# 自作のデータ読み込み&前処理用ライブラリ
from lib.tfidf import TfidfModel
from lib.utils import stems
from lib.utils import stopwords
from lib import utils
import gensim
from pprint import pprint
import matplotlib.pyplot as plt
import datetime
import sys
import re
import numpy as np
from tqdm import tqdm


def load_data_for_segmentation(doc_num):
    print('Interview:',  doc_num)
    # path = './data/segmentation/sentence/interview-text_' + doc_num + '.txt'
    # ans
    path = './data/eval/interview-text_sentence_' + doc_num + '.txt'

    return utils.load_data_for_eval(path)


def load_data_per_interview(doc_num):
    print('Interview:',  doc_num)
    path = './data/interview/interview-text_01-26_' + doc_num + '.txt'

    return utils.load_text(path)


if __name__ == '__main__':
    args = sys.argv
    if 2 <= len(args):
        if not(args[1] == 'sentence' or args[1] == 'segmentation' or args[1] == 'utterance' or args[1] == 'interview'):
            print('Argument is invalid')
            exit()
    else:
        print('Arguments are too sort')
        exit()

    doc_type = args[1]

    doc_num = 'all'
    path = './data/interview/interview-text_01-26_' + doc_num + '.txt'

    if doc_type == 'sentence':
        data = utils.load_data(path)
        # to sentence
        data = utils.to_sentence(data)
        docs = [row[1] for row in data]

    if doc_type == 'utterance':
        data = utils.load_data(path)
        docs = [row[1] for row in data]

    if doc_type == 'interview':
        if doc_num == 'all':
            doc_num = '26'
        docs = []
        for num in range(int(doc_num)):
            num += 1
            if num < 10:
                num = '0' + str(num)
            else:
                num = str(num)
            docs.append(load_data_per_interview(num))

    elif doc_type == 'segmentation':
        if doc_num == 'all':
            doc_num = '26'
        data_arr = []
        for num in range(int(doc_num)):
            num += 1
            if num < 10:
                num = '0' + str(num)
            else:
                num = str(num)
            data_arr.append(load_data_for_segmentation(num))

        # セグメント単位でまとめる
        docs = []
        for data in data_arr:
            tmp_docs = []
            for item in data.items():
                if '_____' in item[1][0]:
                    docs.append('\n'.join(tmp_docs))
                    tmp_docs = []
                else:
                    tmp_docs.extend([item[1][1]])
            docs.append('\n'.join(tmp_docs))

    if doc_num == 'all':
        doc_num = '26'
    doc_num = '01_' + doc_num

    # Params
    no_below = 5
    no_above = 0.5
    keep_n = 100000

    print('===コーパス生成===')
    dictionary = gensim.corpora.Dictionary.load_from_text('./model/tfidf/dict_' + str(no_below) + '_' + str(int(no_above * 100)) + '_' + str(keep_n) + '.dict')
    sw = stopwords()
    docs_for_training = [stems(doc, polish=True, sw=sw) for doc in docs]
    corpus = list(map(dictionary.doc2bow, docs_for_training))
    print(docs[-3:])

    #Metrics for Topic Models
    start = 2
    limit = 30
    step = 1

    coherence_vals = []
    perplexity_vals = []

    for n_topic in tqdm(range(start, limit, step)):
        # LDAモデルの構築
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=n_topic, random_state=0)
        perplexity_vals.append(np.exp2(-lda_model.log_perplexity(corpus)))
        coherence_model_lda = gensim.models.CoherenceModel(model=lda_model, texts=docs_for_training, dictionary=dictionary, coherence='c_v')
        coherence_vals.append(coherence_model_lda.get_coherence())

    # evaluation
    x = range(start, limit, step)

    fig, ax1 = plt.subplots(figsize=(12,5))

    # coherence
    c1 = 'darkturquoise'
    ax1.plot(x, coherence_vals, 'o-', color=c1)
    ax1.set_xlabel('Num Topics')
    ax1.set_ylabel('Coherence', color=c1); ax1.tick_params('y', colors=c1)

    # perplexity
    c2 = 'slategray'
    ax2 = ax1.twinx()
    ax2.plot(x, perplexity_vals, 'o-', color=c2)
    ax2.set_ylabel('Perplexity', color=c2); ax2.tick_params('y', colors=c2)

    # Vis
    ax1.set_xticks(x)
    fig.tight_layout()
    plt.show()
