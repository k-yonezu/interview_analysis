import networkx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class LexRank(object):
    def __init__(self, sent_vecs, *, alpha=0.85, max_iter=100000, threshold=0.1):
        self.sent_vecs = np.array(sent_vecs)
        self.alpha = alpha
        self.max_iter=max_iter
        self.threshold=threshold

    def calc_score(self):
        print('Calc cosine similarity')
        sim_mat = cosine_similarity(self.sent_vecs)
        print(sim_mat)
        linked_rows, linked_cols = np.where(sim_mat > self.threshold)

        # 類似度グラフの生成
        print('Generate graph')
        graph = networkx.DiGraph()
        graph.add_nodes_from(range(self.sent_vecs.shape[0]))
        for i, j in zip(linked_rows, linked_cols):
            if i == j:
                continue
            weight = sim_mat[i,j]
            graph.add_edge(i, j, attr_dict={'weight': weight})

        # PageRank計算
        print('Calc score')
        scores = networkx.pagerank_scipy(graph, alpha=self.alpha, max_iter=self.max_iter)
        print('Done')

        return scores, sim_mat



def lexrank(sent_vecs, dictionary, alpha=0.85, max_iter=100000):
    # comvert to sparse matrix
    sent_vecs_sparse = convert_to_sparse_vector(sent_vecs, len(dictionary.token2id))

    # 文同士のコサイン類似度の計算
    sim_mat = cosine_similarity(sent_vecs_sparse)
    linked_rows, linked_cols = np.where(sim_mat > 0)

    # 類似度グラフの生成
    graph = networkx.DiGraph()
    graph.add_nodes_from(range(sent_vecs_sparse.shape[0]))
    for i, j in zip(linked_rows, linked_cols):
        if i == j:
            continue
        weight = sim_mat[i,j]
        graph.add_edge(i, j, attr_dict={'weight': weight})

    # PageRank計算
    print('Pagerank start')
    scores = networkx.pagerank_scipy(graph, alpha=alpha, max_iter=max_iter)
    print('Done')

    return scores, sim_mat


def convert_to_sparse_vector(sent_vecs, n):
    res = []
    for vec in sent_vecs:
        tmp = np.zeros(n)
        for key, val in vec:
            tmp[key] = val
        res.append(tmp)

    res = np.array(res)
    return res
