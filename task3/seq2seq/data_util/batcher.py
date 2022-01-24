import queue as Queue
import time
from random import shuffle
from threading import Thread

import numpy as np
import tensorflow as tf

import data_util.data as data

import random
random.seed(111) 


class Example(object):

    def __init__(self,args, dialogue, abstract_sentences, vocab):
        start_decoding = vocab.word2id(data.START_DECODING)
        stop_decoding = vocab.word2id(data.STOP_DECODING)
        self.args = args
        # 处理对话文本
        dialogue_words = dialogue.split()
        if len(dialogue_words) > args.max_enc_steps:
            dialogue_words = dialogue_words[:args.max_enc_steps]
        self.enc_len = len(dialogue_words) # 截断之后，padding之前的长度
        self.enc_input = [vocab.word2id(w) for w in dialogue_words] # 字的id组成的列表; 字典以外的字符对应的是 UNK token的id

        # 处理诊疗报告
        abstract = ' '.join(abstract_sentences) # 字符串
        abstract_words = abstract.split() # 字符串组成的列表
        abs_ids = [vocab.word2id(w) for w in abstract_words] # 字的id组成的列表; 字典以外的字符对应的是 UNK token的id

        # 获取解码器的输入和目标序列
        self.dec_input, self.target = self.get_dec_inp_targ_seqs(abs_ids, args.max_dec_steps, start_decoding, stop_decoding)
        self.dec_len = len(self.dec_input)

        if args.pointer_gen:
        # 在Pointer_gen模型中，如果对话内出现OOV，则扩充词典。
            self.enc_input_extend_vocab, self.dialogue_oovs = data.dialogue2ids(dialogue_words, vocab)

            abs_ids_extend_vocab = data.abstract2ids(abstract_words, vocab, self.dialogue_oovs)

            _, self.target = self.get_dec_inp_targ_seqs(abs_ids_extend_vocab, args.max_dec_steps, start_decoding, stop_decoding)

        # 保存原字符串
        self.original_dialogue = dialogue
        self.original_abstract = abstract
        self.original_abstract_sents = abstract_sentences


    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len: # 超过长度，截断
            inp = inp[:max_len]
            target = target[:max_len] 
        else: 
            target.append(stop_id) 
        assert len(inp) == len(target)
        return inp, target


    def pad_decoder_inp_targ(self, max_len, pad_id):
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)


    def pad_encoder_input(self, max_len, pad_id):
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)
        if self.args.pointer_gen:
            while len(self.enc_input_extend_vocab) < max_len:
                self.enc_input_extend_vocab.append(pad_id)


class Batch(object):
    def __init__(self, args,example_list, vocab, batch_size):
        self.args = args
        self.batch_size = batch_size
        self.pad_id = vocab.word2id(data.PAD_TOKEN) 
        self.init_encoder_seq(example_list) 
        self.init_decoder_seq(example_list) 
        self.store_orig_strings(example_list) 


    def init_encoder_seq(self, example_list):
        
        max_enc_seq_len = max([ex.enc_len for ex in example_list])

        for ex in example_list:
            ex.pad_encoder_input(max_enc_seq_len, self.pad_id)

        self.enc_batch = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
        self.enc_lens = np.zeros((self.batch_size), dtype=np.int32)
        self.enc_padding_mask = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.float32)

        for i, ex in enumerate(example_list):
            self.enc_batch[i, :] = ex.enc_input[:]
            self.enc_lens[i] = ex.enc_len
            for j in range(ex.enc_len):
                self.enc_padding_mask[i][j] = 1

        if self.args.pointer_gen:
            self.max_art_oovs = max([len(ex.dialogue_oovs) for ex in example_list])
            self.art_oovs = [ex.dialogue_oovs for ex in example_list]
            self.enc_batch_extend_vocab = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
            for i, ex in enumerate(example_list):
                self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]

    def init_decoder_seq(self, example_list):
        # Pad 
        for ex in example_list:
            ex.pad_decoder_inp_targ(self.args.max_dec_steps, self.pad_id)

        self.dec_batch = np.zeros((self.batch_size, self.args.max_dec_steps), dtype=np.int32)
        self.target_batch = np.zeros((self.batch_size, self.args.max_dec_steps), dtype=np.int32)
        self.dec_padding_mask = np.zeros((self.batch_size, self.args.max_dec_steps), dtype=np.float32)
        self.dec_lens = np.zeros((self.batch_size), dtype=np.int32)

        for i, ex in enumerate(example_list):
            self.dec_batch[i, :] = ex.dec_input[:]
            self.target_batch[i, :] = ex.target[:]
            self.dec_lens[i] = ex.dec_len
            for j in range(ex.dec_len):
                self.dec_padding_mask[i][j] = 1

    def store_orig_strings(self, example_list):
        self.original_dialogues = [ex.original_dialogue for ex in example_list] # list of lists
        self.original_abstracts = [ex.original_abstract for ex in example_list] # list of lists
        self.original_abstracts_sents = [ex.original_abstract_sents for ex in example_list] # list of list of lists


class Batcher(object):
    BATCH_QUEUE_MAX = 100 #  batch_queue 中最大batch数

    def __init__(self, args, data_path, vocab, mode, batch_size, single_pass):
        self.args = args
        self._data_path = data_path
        self._vocab = vocab
        self._single_pass = single_pass
        self.mode = mode
        self.batch_size = batch_size
        # 初始化
        self._batch_queue = Queue.Queue(self.BATCH_QUEUE_MAX)
        self._example_queue = Queue.Queue(self.BATCH_QUEUE_MAX * self.batch_size)

        if single_pass:
            self._num_example_q_threads = 1 
            self._num_batch_q_threads = 1  
            self._bucketing_cache_size = 1 
            self._finished_reading = False 
        else:
            self._num_example_q_threads = 1 #16 # num threads to fill example queue
            self._num_batch_q_threads = 1 #4  # num threads to fill batch queue
            self._bucketing_cache_size = 1 #100 # how many batches-worth of examples to load into cache before bucketing

        # Start the threads that load the queues
        self._example_q_threads = []
        for _ in range(self._num_example_q_threads):
            self._example_q_threads.append(Thread(target=self.fill_example_queue))
            self._example_q_threads[-1].daemon = True
            self._example_q_threads[-1].start()
        self._batch_q_threads = []
        for _ in range(self._num_batch_q_threads):
            self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
            self._batch_q_threads[-1].daemon = True
            self._batch_q_threads[-1].start()

        
        if not single_pass: 
            self._watch_thread = Thread(target=self.watch_threads)
            self._watch_thread.daemon = True
            self._watch_thread.start()

    def next_batch(self):
        if self._batch_queue.qsize() == 0:
            tf.logging.warning('Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i', self._batch_queue.qsize(), self._example_queue.qsize())
            if self._single_pass and self._finished_reading:
                tf.logging.info("Finished reading dataset in single_pass mode.")
                return None

        batch = self._batch_queue.get() 
        return batch

    def fill_example_queue(self):
        input_gen = self.text_generator(data.example_generator(self._data_path, self._single_pass))

        while True:
            try:
                (dialogue, abstract) = next(input_gen) 
                dialogue = dialogue.decode()
                abstract = abstract.decode()
            except StopIteration: 
                tf.logging.info("The example generator for this example queue filling thread has exhausted data.")
                if self._single_pass:
                    tf.logging.info("single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
                    self._finished_reading = True
                    break
                else:
                    raise Exception("single_pass mode is off but the example generator is out of data; error.")

            abstract_sentences = [sent.strip() for sent in data.abstract2sents(abstract)] 
            example = Example(self.args, dialogue, abstract_sentences, self._vocab) 
            self._example_queue.put(example) 

    def fill_batch_queue(self):
        while True:
            if self.mode == 'decode':
                ex = self._example_queue.get()
                b = [ex for _ in range(self.batch_size)]
                self._batch_queue.put(Batch(self.args, b, self._vocab, self.batch_size))
            else:
                inputs = []
                for _ in range(self.batch_size * self._bucketing_cache_size):
                    inputs.append(self._example_queue.get())
                inputs = sorted(inputs, key=lambda inp: inp.enc_len, reverse=True) 

                batches = []
                for i in range(0, len(inputs), self.batch_size):
                    batches.append(inputs[i:i + self.batch_size])
                if not self._single_pass:
                    shuffle(batches)
                for b in batches:  # b 是 Example objects 组成的列表
                    self._batch_queue.put(Batch(self.args,b, self._vocab, self.batch_size))

    def watch_threads(self):
        while True:
            tf.logging.info(
            'Bucket queue size: %i, Input queue size: %i',
            self._batch_queue.qsize(), self._example_queue.qsize())

            time.sleep(60)
            for idx,t in enumerate(self._example_q_threads):
                if not t.is_alive(): # if the thread is dead
                    tf.logging.error('Found example queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_example_queue)
                    self._example_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()
            for idx,t in enumerate(self._batch_q_threads):
                if not t.is_alive(): # if the thread is dead
                    tf.logging.error('Found batch queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_batch_queue)
                    self._batch_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()


    def text_generator(self, example_generator):
        while True:
            e = next(example_generator) # e 是个 tf.Example
            try:
                dialogue_text = e.features.feature['dialogue'].bytes_list.value[0] #  对话文本保存在key 'dialogue'中
                abstract_text = e.features.feature['abstract'].bytes_list.value[0] # 诊疗报告文本保存在key'abstract' 中
            except ValueError:
                tf.logging.error('Failed to get dialogue or abstract from example')
                continue
            if len(dialogue_text)==0: 
                continue
            else:
                yield (dialogue_text, abstract_text)
