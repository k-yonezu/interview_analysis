from lib.utils import stems
from lib.tfidf import TfidfModel
from lib.doc2vec import Doc2Vec
from lib.word2vec import Word2Vec
from lib.text_tiling import TextTiling
from lib import utils
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import datetime
import sys


if __name__ == '__main__':
    args = sys.argv
    if 2 <= len(args):
        if not(args[1] == 'tfidf' or args[1] == 'doc2vec' or args[1] == 'word2vec'):
            print('Argument is invalid')
            exit()
    else:
        print('Arguments are too sort')
        exit()

    model_type = args[1]

    # docs: インタビュー全体
    # 文章番号
    doc_num = '01'
    print('Load data')
    path = './data/interview-text_01-26_' + doc_num + '.txt'
    # path = './data/interview-text_01-26_all.txt'
    data = utils.load_data(path)
    print('Done')

    if model_type == 'tfidf':
        # TFIDFモデル
        model = TfidfModel(no_below=10, no_above=0.1, keep_n=100000)
        model.load_model()
    elif model_type == 'doc2vec':
        model = Doc2Vec(alpha=0.025, min_count=10, vector_size=300, epochs=50, workers=4)
        # model.load_model('./model/doc2vec/doc2vec_' + str(model.vector_size) + '.model')
        # model.load_model('./model/doc2vec/doc2vec_wiki.model')
        model.load_model('./model/doc2vec/updated_doc2vec_300.model')
    elif model_type == 'word2vec':
        model = Word2Vec(alpha=0.025, min_count=10, vector_size=200, epochs=50, workers=4)
        model.load_model('./model/word2vec/word2vec_' + str(model.vector_size) + '.model')
        # model.load_model('./model/word2vec/word2vec_wiki.model')
        # model.load_model('./model/word2vec/updated_word2vec_50.model')
    else:
        print('Invalid model type')
        exit()

    # 発言単位
    docs = [row[1] for row in data]
    label = [row[0] for row in data]
    print(docs[:1])

    # print('===セグメンテーション===')
    # コサイン類似度
    # 可視化
    window_size = 5
    text_tiling = TextTiling(window_size=window_size, p_limit=0.1, a=0.5, model=model)
    res = text_tiling.segment([stems(doc) for doc in docs])

    path_for_ans = './data/interview-text_ans_' + doc_num + '.txt'
    data_for_ans = utils.load_data_for_ans(path_for_ans)
    label_for_ans = [item[0] for item in data_for_ans.items() if '____' in item[1][0]]

    print(res.index.values)
    print(label_for_ans)

    save_path = './result/segmentation/' + model_type + '/evaluation/' + model_type + '_doc_num_' + doc_num +'_window_size_' + str(text_tiling.window_size) + '_' + str(datetime.date.today())

    with open(save_path + '.txt', 'w') as f:
        print(res.index.values, file=f)
        print(label_for_ans, file=f)
