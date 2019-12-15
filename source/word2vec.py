# -*- coding: utf-8 -*-
import h5py
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import gensim
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec


class EpochSaver(CallbackAny2Vec):
    def __init__(self, save_dir):
        self.save_path = save_dir
        self.epoch = 0
        self.epoch_loss = 0
        self.pre_loss = 0
        self.best_loss = np.inf
        self.since = time.time()
        self.history = {'epoch': [], 'loss': []}
        self.state = True

    def on_epoch_end(self, model):
        self.epoch += 1
        self.current_loss = model.get_latest_training_loss()
        self.epoch_loss = self.current_loss - self.pre_loss
        self.pre_loss = self.current_loss

        self.history['epoch'].append(self.epoch)
        self.history['loss'].append(self.epoch_loss)

        time_taken = time.time()- self.since
        print("Epoch %d, loss: %.2f, current loss: %.2f, time: %ds" % (self.epoch, self.epoch_loss, self.current_loss, time_taken))
        if self.best_loss > self.epoch_loss:
            self.best_loss = self.epoch_loss
            print("Better model %s. Best loss: %.6f" % (self.save_path, self.best_loss))
            model.save(self.save_path + 'word2vec.model')
            model.wv.save_word2vec_format(self.save_path + 'model.wv')
        self.since = time.time()


class Gene3Corpus(object):
    def __init__(self, f_name, max_sentence_length=10000, limit=None):
        """
        init function of Gene3Corpus.
        :param f_name: file name
        :param max_sentence_length: max words in sentence
        :param limit: int or None. Clip the file to the first `limit` lines.
        """
        self.f_name = f_name
        self.max_sentence_length = max_sentence_length
        self.limit = limit

    def __iter__(self):
        # the entire corpus is one gigantic line -- there are no sentence marks at all
        # so just split the sequence of tokens arbitrarily: 1 sentence = 1000 tokens
        with gensim.utils.smart_open(self.f_name) as f:
            for line in itertools.islice(f, self.limit):
                line = gensim.utils.to_unicode(line).split()
                i = 0
                while i < len(line):
                    yield line[i: i + self.max_sentence_length]
                    i += self.max_sentence_length

    @classmethod
    def get_vocabulary(cls):
        alpha = ['A', 'C', 'G', 'T']
        vocabulary = []
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    word = alpha[i]+alpha[j]+alpha[k]
                    vocabulary.append(word)
        return [vocabulary]

    @classmethod
    def preprocess(cls):
        alpha = ['A', 'C', 'G', 'T']
        filename = './data/train.mat'
        with h5py.File(filename, 'r') as file:
            one_hot_data = file['trainxdata'][:, :, :]  # shape = (1000, 4, 4400000)
            one_hot_data = np.transpose(one_hot_data, (2, 0, 1))  # shape = (4400000, 1000, 4)

        with open('./data/sequence.fasta', 'w+') as f:
            for sample in tqdm(one_hot_data, desc='Processing...', ascii=True):
                sequence = [np.argmax(i) for i in sample]
                sequence = [alpha[i.item()] for i in sequence]
                sequence = ''.join(sequence)
                sentence = cls.kmer_segment(sequence)
                for i in sentence:
                    f.write(' '.join(i) + '\n')

        print("Process over!")
        return './data/sequence.fasta'

    @staticmethod
    def kmer_segment(sequence):
        sentence_0 = [sequence[i: i+3] for i in range(0, len(sequence)-2, 3)]
        sentence_1 = [sequence[i: i+3] for i in range(1, len(sequence)-2, 3)]
        sentence_2 = [sequence[i: i+3] for i in range(2, len(sequence)-2, 3)]
        return [sentence_0, sentence_1, sentence_2]


def train():
    # 1\ Get the sequence.
    # file = Gene3Corpus.preprocess() # Preprocess
    file = './data/sequence.fasta'
    vocab_list = Gene3Corpus.get_vocabulary()
    sequence_data = Gene3Corpus(file)

    # 2\ Build the Word2vec model.
    model_word2vec = Word2Vec(min_count=1, window=10, size=4, workers=8, batch_words=100000, seed=42)

    # 3\ Build vocabulary.
    since = time.time()
    model_word2vec.build_vocab(vocab_list)
    time_elapsed = time.time() - since
    print('Time to build vocab: {:.0f}min {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # 4\ Training the model
    since = time.time()
    model_word2vec.train(sequence_data, total_examples=4400000 * 3, epochs=20, compute_loss=True,
                         report_delay=1, callbacks=[EpochSaver('./data/')])
    time_elapsed = time.time() - since
    print('Time to train: {:.0f}min {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


def eval():
    model_word2vec = Word2Vec.load('./processing/gene2vec/word2vec.model')
    print(model_word2vec.running_training_loss)
    model_word2vec.get_latest_training_loss()

    # 1\ Vocabulary list, Word Vector list and Similarity Matrix.
    vocab_list = model_word2vec.wv.index2word
    print('Vocabulary List: \n', vocab_list)
    vector_list = model_word2vec.wv.vectors
    print('Word Vector List: \n', vector_list)

    similarity_matrix = []
    for word_1 in vocab_list:
        temp = []
        for word_2 in vocab_list:
            temp.append(model_word2vec.wv.similarity(word_1, word_2))
        similarity_matrix.append(temp)
    print('Similarity Matrix: \n', similarity_matrix)
    np.save('./processing/gene2vec/data/similarity_matrix', np.array(similarity_matrix))

    fig, ax = plt.subplots()
    im = ax.imshow(np.array(similarity_matrix))
    ax.set_xticks(np.arange(len(vocab_list)))
    ax.set_yticks(np.arange(len(vocab_list)))
    ax.set_xticklabels(vocab_list)
    ax.set_yticklabels(vocab_list)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax.set_title("Similarity Matrix")
    fig.tight_layout()
    plt.savefig('./processing/gene2vec/' + 'similarity.png')

    # 2\ Then upload the vecs.tsv and meta.tsv to http://projector.tensorflow.org/, and get the visualization result.
    with open('./processing/gene2vec/vecs.tsv', 'w', encoding='utf-8') as out_v, open('./processing/gene2vec/meta.tsv', 'w', encoding='utf-8') as out_m:
        for word_num in range(len(vocab_list)):
            word = vocab_list[word_num]
            embeddings = vector_list[word_num]
            out_m.write(word + "\n")
            out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")


if __name__ == '__main__':
    train()
    eval()