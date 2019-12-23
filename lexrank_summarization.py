# 自作のデータ読み込み&前処理用ライブラリ
from lib import scraping
from lib.tfidf import TfidfModel
from lib.doc2vec import Doc2Vec
from lib.summarize import summarize
from lib.utils import stems
from lib import utils
import datetime
import sys


def validate_args(args):
    if 2 <= len(args):
        if not(args[1] == 'tfidf' or args[1] == 'doc2vec'):
            print('Argument is invalid')
            exit()

        if not(args[2] == 'sentence' or args[2] == 'utterance' or args[2] == 'segmentation'):
            print('Argument is invalid')
            exit()

        if not(args[3] == 'mmr' or args[3] == 'normal'):
            print('Argument is invalid')
            exit()
    else:
        print('Arguments are too sort')
        exit()

    return args[1], args[2], args[3]


def load_data_for_segmentation(doc_num):
    print('Interview:',  doc_num)
    path = './data/segmentation/segmentation_interview-text_' + doc_num + '.txt'

    return utils.load_data_segment(path)


if __name__ == '__main__':

    model_type, sum_type, sort_type = validate_args(sys.argv)

    # ハイパーパラメータ
    threshold = 0

    # docs: インタビュー全体
    print('Load data')
    doc_num = 'all'
    path = './data/interview/interview-text_01-26_' + doc_num + '.txt'
    # path = './data/interview_tfidf_doc_all_window_size_10_2019-10-23.txt'

    # 要約する単位 文 or 発言
    if sum_type == 'sentence':
        data = utils.load_data(path)
        # to sentence
        data = utils.to_sentence(data)
        docs = [row[1] for row in data]
    if sum_type == 'utterance':
        data = utils.load_data(path)
        docs = [row[1] for row in data]
    if sum_type == 'segmentation':
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

        doc_num = '01_' + doc_num

    print('Done')

    # for sum
    print(docs[:3])

    if model_type == 'tfidf':
        # GensimのTFIDFモデルを用いた文のベクトル化
        tfidf = TfidfModel(no_below=10, no_above=0.1, keep_n=100000)
        tfidf.load_model()
        sent_vecs = tfidf.to_vector([stems(doc) for doc in docs])

    elif model_type == 'doc2vec':
        # ===Doc2Vec===
        doc2vec = Doc2Vec(alpha=0.025, min_count=10, vector_size=300, epochs=50, workers=4)
        model_path = './model/doc2vec/doc2vec_' + str(doc2vec.vector_size) + '.model'
        doc2vec.load_model(model_path)
        sent_vecs = doc2vec.to_vector(([stems(doc) for doc in docs]))

    else:
        print('Invalid model type')
        exit()

    # 表示
    print('===要約===')
    # 要約
    indexes = summarize(docs, sent_vecs, sort_type=sort_type, sent_limit=10, threshold=threshold)
    docs_summary = [docs[i] for i in indexes]

    with open('./result/summary/' + model_type + '/doc_num_' + doc_num + '_' + sum_type + '_' + sort_type + '_' + str(datetime.date.today()) + '.txt', 'w') as f:
        if model_type == 'tfidf':
            print("no_below: " + str(tfidf.no_below) + ", no_above: " + str(tfidf.no_above) + ", keep_n: " + str(tfidf.keep_n) + ", threshold: " + str(threshold), file=f)
        if model_type == 'doc2vec':
            print("doc2vec model: " + model_path + ", threshold: " + str(threshold), file=f)
        for i, docs in enumerate(docs_summary):
            print(str(i) + ': ' + docs.strip(), file=f)
