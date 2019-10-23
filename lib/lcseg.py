from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import copy
import math


class LexicalCohesionSegmentation(object):
    def __init__(self, *, window_size=2, hiatus=5, p_limit=0.1, a=0.5, model=False):
        self.window_size = window_size
        self.p_limit = p_limit
        self.a = a
        self.model = model
        self.sim_arr = pd.Series()
        self.p_arr = pd.Series()
        self.chain_arr = pd.Series()
        self.hiatus = hiatus

    def segment(self, docs):
        self.generate_chain_arr(docs)
        self.calc_sim(docs)

        r, m, l = None, None, None
        for i in range(len(self.sim_arr)):
            if i + 1 > len(self.sim_arr) - 1:
                break
            val, val_next = self.sim_arr[i], self.sim_arr[i + 1]
            if not(r) and val > val_next:
                r = val
            if r:
                if not(m) and val < val_next:
                    m = val
                if m and not(l) and val > val_next:
                    l = val

            if r and m and l:
                p = (l + r - 2 * m) * 0.5
                if p >= self.p_limit:
                    self.p_arr[self.sim_arr.index.values[i]] = p
                r = l
                m, l = None, None

        return self.p_arr[self.p_arr > (self.p_arr.mean() - self.a * self.p_arr.std())]

    def calc_sim(self, docs):
        # 窓を動かしながらコサイン類似度を計算する
        for i in range(self.window_size, len(docs)):
            left_start = i - self.window_size
            left_end = left_start + self.window_size - 1
            right_start = i
            right_end = right_start + self.window_size - 1

            if right_end > len(docs) - 1:
                break

            left_window = copy.deepcopy(docs[left_start])
            for arr in docs[left_start + 1:left_end + 1]:
                left_window.extend(arr)

            right_window = copy.deepcopy(docs[right_start])
            for arr in docs[right_start + 1:right_end + 1]:
                right_window.extend(arr)

            # GensimのTFIDFモデルを用いた文のベクトル化
            self.sim_arr[str(i - 0.5)] = self.lcf(left_window, right_window, i)

    def lcf(self, left, right, index):
        chain_word = []
        vec_left = np.zeros(len(self.model.dictionary), dtype="float32")
        vec_right = np.zeros(len(self.model.dictionary), dtype="float32")
        for word in left:
            if word in self.chain_arr:
                for e in self.chain_arr[word]:
                    if index >= e['chain'][0] and index <= e['chain'][-1]:
                        vec_left[self.model.dictionary.token2id[word]] = e['score']

        for word in right:
            if word in self.chain_arr:
                for e in self.chain_arr[word]:
                    if index >= e['chain'][0] and index <= e['chain'][-1]:
                        vec_right[self.model.dictionary.token2id[word]] = e['score']

        return cosine_similarity([vec_right], [vec_left])[0][0]


    def generate_chain_arr(self, docs):
        for word in self.model.dictionary.token2id.keys():
            chain = []
            tmp_chain = []
            for i in range(len(docs)):
                if word in docs[i]:
                    if len(tmp_chain) > 0 and i - tmp_chain[-1] > self.hiatus:
                        if len(tmp_chain) > 1:
                            # Calc socre
                            chain.append({'score': self.calc_chain_score(docs, tmp_chain, word), 'chain': tmp_chain})
                        tmp_chain = []
                    tmp_chain.append(i)

            if len(tmp_chain) > 1:
                # Calc socre
                chain.append({'score': self.calc_chain_score(docs, tmp_chain, word), 'chain': tmp_chain})

            if len(chain) > 0:
                # Calc socre
                self.chain_arr[word] = chain

    def calc_chain_score(self, docs, chain, word):
        doc = copy.deepcopy(docs[chain[0]])
        for arr in docs[chain[0] + 1:chain[-1] + 1]:
            doc.extend(arr)
        freq = len([w for w in doc if w == word]) / len(doc)

        return freq * math.log(len(docs) / len(chain))
