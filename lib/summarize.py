from lib.sort import Sort
from lib.lexrank import LexRank


def summarize(docs, sent_vecs, *, sort_type='mmr', sent_limit=10, threshold=0):
    """
    use_mmr: 要約実行時にMMR(冗長性排除)を行うか否か
    sent_limit: XX文まで抽出する
    """

    # LexRank算出 → ソート
    lexrank = LexRank(sent_vecs, threshold=threshold)
    sentence_scores, sim_mat = lexrank.calc_score()

    sort = Sort(docs, sentence_scores, sim_mat, sent_limit=sent_limit)

    if sort_type == 'mmr':
        indexes = sort.mmr_sort()

    if sort_type == 'normal':
        indexes = sort.normal_sort()

    # 抽出
    print(indexes)
    summary_sents = [docs[i] for i in indexes]

    return summary_sents
