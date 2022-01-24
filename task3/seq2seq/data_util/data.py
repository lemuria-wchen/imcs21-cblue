import glob
import random
import struct
import csv
from tensorflow.core.example import example_pb2

# <s> 和 </s> 将诊疗报告中的每一个部分分隔开
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]' # 占位符，用在编码器输入、解码器输入和参考诊疗报告中
UNKNOWN_TOKEN = '[UNK]' # 表示未录入的字典的字符
START_DECODING = '[START]' # 表示解码器输入的开始
STOP_DECODING = '[STOP]' # 未被截断的目标序列的结尾

#  <s>, </s>, [PAD], [UNK], [START], [STOP] 不应当出现在字典中


class Vocab(object):
    def __init__(self, vocab_file, max_size):
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0 # 追踪Vocab的总数

        # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
        for w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        with open(vocab_file, 'r',encoding='utf-8') as vocab_f:
            for line in vocab_f:
                pieces = line.split()
                if len(pieces) != 2:
                    print ('Warning: incorrectly formatted line in vocabulary file: %s\n' % line)
                    continue
                w = pieces[0]
                if w in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                    raise Exception('<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
                if w in self._word_to_id:
                    raise Exception('Duplicated word in vocabulary file: %s' % w)
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if max_size != 0 and self._count >= max_size:
                    print ("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self._count))
                    break

        print ("Finished constructing vocabulary of %i total words. Last word added: %s" % (self._count, self._id_to_word[self._count-1]))

    def word2id(self, word):
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def size(self):
        return self._count

    def write_metadata(self, fpath):
        print ("Writing word embedding metadata file to %s..." % (fpath))
        with open(fpath, "w") as f:
            fieldnames = ['word']
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
            for i in xrange(self.size()):
                writer.writerow({"word": self._id_to_word[i]})


def example_generator(data_path, single_pass):
    while True:
        filelist = glob.glob(data_path) 
        assert filelist, ('Error: Empty filelist at %s' % data_path) 
        if single_pass:
            filelist = sorted(filelist)
        else:
            random.shuffle(filelist)
        for f in filelist:
            reader = open(f, 'rb')
            while True:
                len_bytes = reader.read(8)
                if not len_bytes: break 
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                yield example_pb2.Example.FromString(example_str)
        if single_pass:
            print ("example_generator completed reading all datafiles. No more data.")
            break


def dialogue2ids(dialogue_words, vocab):
    ids = []
    oovs = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in dialogue_words:
        i = vocab.word2id(w)
        if i == unk_id: # If w is OOV
            if w not in oovs: # Add to list of OOVs
                oovs.append(w)
            oov_num = oovs.index(w) # This is 0 for the first dialogue OOV, 1 for the second dialogue OOV...
            ids.append(vocab.size() + oov_num) # This is e.g. 3000 for the first dialogue OOV, 50001 for the second...
        else:
            ids.append(i)
    return ids, oovs


def abstract2ids(abstract_words, vocab, dialogue_oovs):
    ids = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in abstract_words:
        i = vocab.word2id(w)
        if i == unk_id: # If w is an OOV word
            if w in dialogue_oovs: # If w is an in-dialogue OOV
                vocab_idx = vocab.size() + dialogue_oovs.index(w) # Map to its temporary dialogue OOV number
                ids.append(vocab_idx)
            else: # If w is an out-of-dialogue OOV
                ids.append(unk_id) # Map to the UNK token id
        else:
            ids.append(i)
    return ids


def outputids2words(id_list, vocab, dialogue_oovs):
    words = []
    for i in id_list:
        try:
            w = vocab.id2word(i) # might be [UNK]
        except ValueError as e: # w is OOV
            assert dialogue_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
            dialogue_oov_idx = i - vocab.size()
            try:
                w = dialogue_oovs[dialogue_oov_idx]
            except ValueError as e: 
                raise ValueError('Error: model produced word ID %i which corresponds to dialogue OOV %i but this example only has %i dialogue OOVs' % (i, dialogue_oov_idx, len(dialogue_oovs)))
        words.append(w)
    return words


def abstract2sents(abstract):
    cur = 0
    sents = []
    while True:
        try:
            start_p = abstract.index(SENTENCE_START, cur)
            end_p = abstract.index(SENTENCE_END, start_p + 1)
            cur = end_p + len(SENTENCE_END)
            sents.append(abstract[start_p+len(SENTENCE_START):end_p])
        except ValueError as e: # no more sentences
            return sents


def show_art_oovs(dialogue, vocab):
    unk_token = vocab.word2id(UNKNOWN_TOKEN)
    words = dialogue.split(' ')
    words = [("__%s__" % w) if vocab.word2id(w)==unk_token else w for w in words]
    out_str = ' '.join(words)
    return out_str


def show_abs_oovs(abstract, vocab, dialogue_oovs):
    unk_token = vocab.word2id(UNKNOWN_TOKEN)
    words = abstract.split(' ')
    new_words = []
    for w in words:
        if vocab.word2id(w) == unk_token: # w is oov
            if dialogue_oovs is None: # baseline mode
                new_words.append("__%s__" % w)
            else: # pointer-generator mode
                if w in dialogue_oovs:
                    new_words.append("__%s__" % w)
                else:
                    new_words.append("!!__%s__!!" % w)
        else: # w is in-vocab word
            new_words.append(w)
    out_str = ' '.join(new_words)
    return out_str
