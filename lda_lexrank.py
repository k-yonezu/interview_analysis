from lib.tfidf import TfidfModel
from lib.utils import stems
from lib.utils import stopwords
from lib import utils
import gensim
from pprint import pprint
import datetime
import sys
import re
import os
from lib.summarize import summarize


def load_data_for_segmentation(doc_num, *, ans=False):
    print('Interview:',  doc_num)
    path = './data/segmentation/sentence/interview-text_' + doc_num + '.txt'
    # path = './data/segmentation/utterance/interview-text_' + doc_num + '.txt'
    if ans:
        print('ans')
        path = './data/eval/interview-text_sentence_' + doc_num + '.txt'

    return utils.load_data_segment(path)


# def get_docs_topic(docs, prob_arr, topic_N):
#     topic_docs = {}
#     for num in range(topic_N):
#         topic_docs[num] = []
#         for i in range(len(prob_arr)):
#             prob = prob_arr[i]
#             tmp = [val for topic, val in prob if topic == num]
#             if tmp:
#                 topic_docs[num].append(docs[i])

#     return topic_docs


def get_docs_topic(docs, prob_arr, topic_N):
    topic_docs = {}
    for i in range(len(prob_arr)):
        prob = prob_arr[i]
        arr = [x[1] for x in prob]
        topic = prob[arr.index(max(arr))][0]
        print(prob)
        print(topic)
        if topic not in topic_docs.keys():
            topic_docs[topic] = []
        topic_docs[topic].append(docs[i])

    return topic_docs


def lexrank(original_docs, topic, dictionary, dir='./result/lda/summary/'):
    docs = []
    for doc in original_docs:
        tmp_docs = []
        for speaker, remark in doc:
            tmp_docs.append(remark)
        docs.append('\n'.join(tmp_docs))

    # for training
    tfidf = TfidfModel(no_below=0, no_above=1.0, keep_n=100000)
    docs_for_training = [stems(doc) for doc in docs]
    tfidf.train(docs_for_training)
    sent_vecs = tfidf.to_vector(docs_for_training)

    # for dict
    sw = stopwords()
    docs_for_dict = [stems(doc, polish=True, sw=sw) for doc in docs]
    corpus = list(map(dictionary.doc2bow, docs_for_dict))

    # 表示
    print('===要約===')
    # 要約
    indexes = summarize(docs, sent_vecs, sort_type='normal', sent_limit=10, threshold=0.1)
    docs_summary = [original_docs[i] for i in indexes]
    probs_summary = [lda[corpus[i]] for i in indexes]

    if not(os.path.exists(dir)):
        os.makedirs(dir)
    with open(dir + '/topic_' + str(topic+1) + '.txt', 'w') as f:
        i = 0
        for docs, prob in zip(docs_summary, probs_summary):
            i += 1
            print("-"*80, file=f)
            print(str(i) + ':', file=f)
            print([(t+1, p) for t, p in prob], file=f)
            for speaker, remark in docs:
                print(speaker + '　' + remark, file=f)


if __name__ == '__main__':
    args = sys.argv
    if 2 <= len(args):
        if not(args[1] == 'sentence' or args[1] == 'segmentation' or args[1] == 'utterance' or args[1] == 'segmentation/ans'):
            print('Argument is invalid')
            exit()
    else:
        print('Arguments are too sort')
        exit()

    doc_type = args[1]

    doc_num = 'all'
    ans = False
    if doc_type == 'segmentation/ans':
        ans = True

    if doc_num == 'all':
        doc_num = '26'
    data_arr = []
    for num in range(int(doc_num)):
        num += 1
        if num < 10:
            num = '0' + str(num)
        else:
            num = str(num)
        data_arr.append(load_data_for_segmentation(num, ans=ans))

    original_docs = []
    for data in data_arr:
        tmp_docs = []
        for item in data.items():
            if '_____' in item[1][0]:
                original_docs.append(tmp_docs)
                tmp_docs = []
            else:
                tmp_docs.append((item[1][0], item[1][1]))
        original_docs.append(tmp_docs)

    # セグメント単位でまとめる
    docs = []
    for arr in original_docs:
        tmp_docs = []
        for speaker, remark in arr:
            tmp_docs.append(remark)
        docs.append('\n'.join(tmp_docs))

    if doc_num == 'all':
        doc_num = '26'
    doc_num = '01_' + doc_num

    # Params
    no_below = 3
    no_above = 0.8
    keep_n = 100000
    topic_N = 9
    sw = stopwords()
    docs_for_training = [stems(doc, polish=True, sw=sw) for doc in docs]

    print('===コーパス生成===')
    dictionary = gensim.corpora.Dictionary(docs_for_training)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    # Load
    corpus = list(map(dictionary.doc2bow, docs_for_training))
    print(docs[-3:])

    # Load lda
    model_name = 'doc_num_' + doc_num + '_topic_' + str(topic_N)
    lda = gensim.models.ldamodel.LdaModel.load('./model/lda/' + doc_type + '/' + model_name + '.model')

    # Calc
    prob_arr = [lda[doc] for doc in corpus]
    topic_docs = get_docs_topic(original_docs, prob_arr, topic_N)
    print(topic_docs.keys())

    dir = './result/lda/summary/'
    if ans:
        dir += 'ans/'
    # for topic
    for topic, docs in topic_docs.items():
        lexrank(docs, topic, dictionary, dir=dir+model_name)
