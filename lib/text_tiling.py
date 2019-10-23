from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import copy


class TextTiling(object):
    def __init__(self, *, window_size=2, p_limit=0.1, a=0.5, model=False):
        self.window_size = window_size
        self.p_limit = p_limit
        self.a = a
        self.model = model
        self.sim_arr = pd.Series()
        self.p_arr = pd.Series()

    def segment(self, docs):
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
            self.sim_arr[str(i - 0.5)] = cosine_similarity(self.model.to_vector([left_window]), self.model.to_vector([right_window]))[0][0]



