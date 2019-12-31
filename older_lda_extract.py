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


def load_data_for_segmentation(doc_num, *, ans=False):
    print('Interview:',  doc_num)
    path = './data/segmentation/sentence/interview-text_' + doc_num + '.txt'
    # path = './data/segmentation/utterance/interview-text_' + doc_num + '.txt'
    if ans:
        print('ans')
        path = './data/eval/interview-text_sentence_' + doc_num + '.txt'

    return utils.load_data_segment(path)


def write_topic_probs(original_docs, prob_arr, topic_N, dir='./result/lda/extracted/'):
    if not(os.path.exists(dir)):
        os.makedirs(dir)
    for num in range(topic_N):
        with open(dir + '/topic_' + str(num + 1) + '.txt', 'w') as f:
            probs = []
            for arr in prob_arr:
                tmp = [val for topic, val in arr if topic == num]
                if tmp:
                    probs.extend(tmp)
                else:
                    probs.extend([0])
            probs = dict(enumerate(probs))
            sorted_probs = sorted(probs.items(), key=lambda x:x[1], reverse=True)
            print(sorted_probs[:5])
            # print(docs[sorted_probs[0][0]])
            i = 0
            for docs in [original_docs[i] for i, prob in sorted_probs[:10]]:
                i += 1
                print("-"*80, file=f)
                print(str(i) + ':', file=f)
                for speaker, remark in docs:
                    print(speaker + '　' + remark, file=f)


if __name__ == '__main__':
    args = sys.argv
    if 2 <= len(args):
        if not(args[1] == 'sentence' or args[1] == 'segmentation' or args[1] == 'utterance' or args[1] == 'segmentation/ans'):
            print('Argument is invalid')
            exit()
        if not(args[1] == 'segmentation' or args[1] == 'segmentation/ans'):
            print('Argument is invalid')
            exit()
    else:
        print('Arguments are too sort')
        exit()

    model_doc_type = args[1]
    doc_type = args[2]

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
    no_above = 0.5
    keep_n = 100000
    topic_N = 5
    sw = stopwords()
    docs_for_training = [stems(doc, polish=True, sw=sw) for doc in docs]

    print('===コーパス生成===')
    dictionary = gensim.corpora.Dictionary(docs_for_training)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    # Load
    # dictionary = gensim.corpora.Dictionary.load_from_text('./model/tfidf/dict_' + str(no_below) + '_' + str(int(no_above * 100)) + '_' + str(keep_n) + '.dict')
    corpus = list(map(dictionary.doc2bow, docs_for_training))
    print(docs[-3:])

    # Load lda
    model_name = 'doc_num_' + doc_num + '_topic_' + str(topic_N)
    lda = gensim.models.ldamodel.LdaModel.load('./model/lda/' + model_doc_type + '/' + model_name + '.model')

    # Calc
    prob_arr = [lda[doc] for doc in corpus]
    print(prob_arr[:5])
    dir = './result/lda/extracted/model_doc_type_' + model_doc_type + '/' + 'doc_type_' + doc_type + '/' + model_name
    write_topic_probs(original_docs, prob_arr, topic_N, dir=dir)
    print(prob_arr[720])
