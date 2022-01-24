import os
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup

from utils import MODEL_CLASSES, compute_metrics, get_seq_labels

logger = logging.getLogger(__name__)


class Trainer(object):
    """模型训练"""
    def __init__(self, args, train_dataset=None, dev_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset

        self.seq_label_lst = get_seq_labels(args)

        # pad标签
        self.pad_token_label_id = args.ignore_index

        # (BertConfig, NERBERT, BertTokenizer)
        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]

        # 加载模型配置和模型
        print(args.model_name_or_path)
        # 在线下载模型
        self.config = self.config_class.from_pretrained(args.model_name_or_path, finetuning_task=args.task)
        self.model = self.model_class.from_pretrained(args.model_name_or_path,
                                                      config=self.config,
                                                      args=args,
                                                      seq_label_lst=self.seq_label_lst)
        # # 从已下载bert模型中加载
        # self.config = self.config_class.from_pretrained('./bert-base-chinese/bert-base-chinese-config.json',finetuning_task=args.task)
        # self.model = self.model_class.from_pretrained('./bert-base-chinese/bert-base-chinese-pytorch_model.bin',
        #                                                 config=self.config,
        #                                               args=args,
        #                                               seq_label_lst=self.seq_label_lst)

        # GPU / CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
        """训练"""
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # 设置optimizer、linear warmup、decay
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU / CPU

                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'seq_labels_ids': batch[3]}

                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # 更新学习率schedule
                    self.model.zero_grad()
                    global_step += 1

                    # 训练一定次数后进行验证和模型保存
                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        self.evaluate("dev")

                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        self.save_model()

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        """验证"""
        if mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        seq_preds = None
        out_seq_labels_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'seq_labels_ids': batch[3]}
                outputs = self.model(**inputs)
                tmp_eval_loss, (seq_logits) = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # 预测
            if seq_preds is None:
                if self.args.use_crf:
                    seq_preds = np.array(self.model.crf.decode(seq_logits))
                else:
                    seq_preds = seq_logits.detach().cpu().numpy()

                out_seq_labels_ids = inputs["seq_labels_ids"].detach().cpu().numpy()
            else:
                if self.args.use_crf:
                    seq_preds = np.append(seq_preds, np.array(self.model.crf.decode(seq_logits)), axis=0)
                else:
                    seq_preds = np.append(seq_preds, seq_logits.detach().cpu().numpy(), axis=0)

                out_seq_labels_ids = np.append(out_seq_labels_ids, inputs["seq_labels_ids"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        # 结果
        if not self.args.use_crf:
            seq_preds = np.argmax(seq_preds, axis=2)
        seq_label_map = {i: label for i, label in enumerate(self.seq_label_lst)}
        out_seq_label_list = [[] for _ in range(out_seq_labels_ids.shape[0])]
        seq_preds_list = [[] for _ in range(out_seq_labels_ids.shape[0])]

        for i in range(out_seq_labels_ids.shape[0]):
            for j in range(out_seq_labels_ids.shape[1]):
                if out_seq_labels_ids[i, j] != self.pad_token_label_id:
                    out_seq_label_list[i].append(seq_label_map[out_seq_labels_ids[i][j]])
                    seq_preds_list[i].append(seq_label_map[seq_preds[i][j]])

        # with open('results_seq_true.txt','w',encoding='utf8') as f:
        #     for i in range(len(out_seq_label_list)):
        #         f.write(','.join(list(out_seq_label_list[i]))+'\r\n')
        # with open('results_seq_pred.txt','w',encoding='utf8') as f:
        #     for i in range(len(seq_preds_list)):
        #         f.write(','.join(list(seq_preds_list[i]))+'\r\n')

        # 计算结果
        total_result = compute_metrics(seq_preds_list, out_seq_label_list)
        results.update(total_result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        return results

    def save_model(self):
        """模型保存"""
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.args.model_dir)

        torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        """模型读取"""
        print('============================模型是：',self.model_class)
        print(self.args)
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model = self.model_class.from_pretrained(self.args.model_dir,
                                                          args=self.args,
                                                          seq_label_lst=self.seq_label_lst)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")
