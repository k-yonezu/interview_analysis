# 自作のデータ読み込み&前処理用ライブラリ
from lib.utils import stems
from lib.utils import stopwords
from lib import utils
import gensim
from pprint import pprint
import datetime
import sys
import re


def load_data_for_segmentation(doc_num):
    print('Interview:',  doc_num)
    path = './data/segmentation/segmentation_interview-text_' + doc_num + '.txt'
    # path = './data/eval/interview-text_eval_' + doc_num + '.txt'

    return utils.load_data_for_eval(path)


if __name__ == '__main__':
    args = sys.argv
    if 2 <= len(args):
        if not(args[1] == 'sentence' or args[1] == 'segmentation' or args[1] == 'docs'):
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

    if doc_type == 'docs':
        data = utils.load_data(path)
        docs = [row[1] for row in data]

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
    topic_N = 8

    dictionary = gensim.corpora.Dictionary.load_from_text('./model/tfidf/dict_' + str(no_below) + '_' + str(int(no_above * 100)) + '_' + str(keep_n) + '.dict')
    sw = stopwords()
    corpus = list(map(dictionary.doc2bow, [stems(doc, polish=True, sw=sw) for doc in docs]))
    print(docs[-3:])

    # LDAモデルの構築
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=topic_N, id2word=dictionary, random_state=0)

    # モデルのsave
    lda.save('./model/lda/' + doc_type +'/' + 'topic_' + str(topic_N) + '.model')

    with open('./result/clustering/lda/' + doc_type +'/' + 'doc_num_' + doc_num + '_topic_' + str(topic_N) + '_' + str(datetime.date.today()) + '.txt', 'w') as f:
        for i in range(topic_N):
            print("\n", file=f)
            print("="*80, file=f)
            print("TOPIC {0}\n".format(i), file=f)
            topic = lda.show_topic(i, topn=30)
            for t in topic:
                print("{0:20s}{1}".format(t[0], t[1]), file=f)

    # for topic in lda.show_topics(-1, num_words=20):
    #     print(topic)
    # # セグメント単位で入れた場合の処理
    # target_record = 5 # 分析対象のドキュメントインデックス
    # print(docs[5])

    # for topics_per_document in lda[corpus[target_record]]:
    #     print(topics_per_document[0], topics_per_document[1])
