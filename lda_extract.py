from lib.tfidf import TfidfModel
from lib.utils import stems
from lib.utils import stopwords
from lib import utils
import gensim
from pprint import pprint
import datetime
import sys
import re


def load_data_for_segmentation(doc_num, *, ans=False):
    print('Interview:',  doc_num)
    path = './data/segmentation/sentence/interview-text_' + doc_num + '.txt'
    # path = './data/segmentation/utterance/interview-text_' + doc_num + '.txt'
    if ans:
        print('a')
        path = './data/eval/interview-text_sentence_' + doc_num + '.txt'

    return utils.load_data_for_eval(path)


def write_topic_probs(docs, prob_arr, model_type, doc_type, doc_num, topic_N):
    with open('./result/sorted/lda/' + 'model_type_' + model_type + '/' + 'doc_type_' + doc_type + '/' + 'doc_num_' + doc_num + '_topic_' + str(topic_N) + '_' + str(datetime.date.today()) + '.txt', 'w') as f:
        for num in range(topic_N):
            probs = []
            for arr in prob_arr:
                tmp = [val for topic, val in arr if topic == num]
                if tmp:
                    probs.extend(tmp)
                else:
                    probs.extend([0])
            probs = dict(enumerate(probs))
            sorted_probs = sorted(probs.items(), key=lambda x:x[1], reverse=True)
            print("\n", file=f)
            print("="*80, file=f)
            print('Topic:', num + 1, file=f)
            i = 0
            print(sorted_probs[:5])
            # print(docs[sorted_probs[0][0]])
            for doc in [docs[e[0]] for e in sorted_probs[:5]]:
                i += 1
                print('---------------------------------------', file=f)
                print(str(i) + ':', file=f)
                print(doc, file=f)


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

    model_type = args[1]
    doc_type = args[2]

    doc_num = 'all'
    # path = './data/interview/interview-text_01-26_' + doc_num + '.txt'
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
    no_below = 1
    no_above = 0.5
    keep_n = 100000
    topic_N = 8
    sw = stopwords()
    docs_for_training = [stems(doc, polish=True, sw=sw) for doc in docs]

    print('===コーパス生成===')
    tfidf = TfidfModel(no_below=no_below, no_above=no_above, keep_n=keep_n)
    tfidf.train(docs_for_training)
    dictionary = tfidf.dictionary
    corpus = tfidf.corpus

    # dictionary = gensim.corpora.Dictionary.load_from_text('./model/tfidf/dict_' + str(no_below) + '_' + str(int(no_above * 100)) + '_' + str(keep_n) + '.dict')
    # corpus = list(map(dictionary.doc2bow, docs_for_training))
    print(docs[-3:])

    # Load lda
    lda = gensim.models.ldamodel.LdaModel.load('./model/lda/' + model_type + '/topic_' + str(topic_N) + '.model')

    # Calc
    prob_arr = [lda[e] for e in corpus]
    # print(res)
    print(prob_arr[:5])
    write_topic_probs(docs, prob_arr, model_type, doc_type, doc_num, topic_N)
    print(prob_arr[720])
