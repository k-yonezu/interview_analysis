import gensim
import numpy as np

class TfidfModel(object):
    '''
    TFIDFモデル、辞書などを保持

    Attributes
    ----------
    self.dictionary : dict
        辞書
    self.corpus : list
        コーパス
    self.model : model
        tfidfモデル
    '''

    def __init__(self, *, no_below=10, no_above=0.05, keep_n=10000):
        '''
        Parameters
        ----------
        docs : list
            対象の文章
        no_below : int
            XX回以下しか出てこない単語は無視
        no_above : int
            頻出単語も無視
        keep_n : int
            使用単語数に上限設定
        '''

        self.no_below = no_below  # XX回以下しか出てこない単語は無視
        self.no_above = no_above  # 頻出単語も無視
        self.keep_n = keep_n  # 使用単語数に上限設定
        self.dictionary = None
        self.model = None
        self.corpus = None

    def load_model(self):
        self.dictionary = gensim.corpora.Dictionary.load_from_text('./model/tfidf/dict_' + str(self.no_below) + '_' + str(int(self.no_above * 100)) + '_' + str(self.keep_n) + '.dict')
        self.corpus = gensim.corpora.MmCorpus('./model/tfidf/corpus_' + str(self.no_below) + '_' + str(int(self.no_above * 100)) + '_' + str(self.keep_n) + '.mm')
        self.model = gensim.models.TfidfModel(self.corpus)

    # モデル生成
    def train(self, docs):
        self.dictionary = gensim.corpora.Dictionary(docs)
        self.dictionary.filter_extremes(no_below=self.no_below, no_above=self.no_above, keep_n=self.keep_n)
        self.corpus = list(map(self.dictionary.doc2bow, docs))
        self.model = gensim.models.TfidfModel(self.corpus)

    def save_model(self, dir='./model/tfidf/'):
        # 保存
        self.dictionary.save_as_text(dir + 'dict_' + str(self.no_below) + '_' + str(int(self.no_above * 100)) + '_' + str(self.keep_n) + '.dict')
        gensim.corpora.MmCorpus.serialize(dir + 'corpus_' + str(self.no_below)  + '_' + str(int(self.no_above * 100)) + '_' + str(self.keep_n) + '.mm', self.corpus)

    # GensimのTFIDFモデルを用いた文のベクトル化
    def to_vector(self, docs):
        sparse = []
        sent_vecs = [self.model[self.dictionary.doc2bow(doc)] for doc in docs]
        for vec in sent_vecs:
            tmp = np.zeros(len(self.dictionary), dtype="float32")
            for key, val in vec:
                tmp[key] = val
            sparse.append(tmp.tolist())

        return sparse
