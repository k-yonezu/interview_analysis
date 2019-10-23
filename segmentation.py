from lib.utils import stems
from lib.tfidf import TfidfModel
from lib.doc2vec import Doc2Vec
from lib.word2vec import Word2Vec
from lib.text_tiling import TextTiling
from lib.lcseg import LexicalCohesionSegmentation
from lib import utils
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import datetime
import sys


if __name__ == '__main__':
    eval = False
    args = sys.argv
    if 2 <= len(args):
        if not(args[1] == 'tfidf' or args[1] == 'doc2vec' or args[1] == 'word2vec'):
            print('Argument is invalid')
            exit()

        if not(args[2] == 'sentence' or args[2] == 'docs'):
            print('Argument is invalid')
            exit()

        if not(args[3] == 'text_tiling' or args[3] == 'lcseg'):
            print('Argument is invalid')
            exit()
        if args[-1] == 'eval':
            eval = True
    else:
        print('Arguments are too sort')
        exit()

    model_type, doc_type, segmentation_type = args[1], args[2], args[3]

    # docs: インタビュー全体
    # 文章番号
    doc_num = '01'
    print('Load data')
    path = './data/interview-text_01-26_' + doc_num + '.txt'
    # path = './data/interview-text_01-26_all.txt'
    data = utils.load_data(path)
    if doc_type == 'sentence':
        data = utils.to_sentence(data)
    print('Done')

    if model_type == 'tfidf':
        # TFIDFモデル
        model = TfidfModel(no_below=2, no_above=0.1, keep_n=100000)
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

    print('===セグメンテーション===')
    # コサイン類似度
    # 可視化
    window_size = 20

    if segmentation_type == 'text_tiling':
        segmentation_model = TextTiling(window_size=window_size, p_limit=0.1, a=0.5, model=model)
    elif segmentation_type == 'lcseg':
        segmentation_model = LexicalCohesionSegmentation(window_size=window_size, hiatus=10, p_limit=0.1, a=0.5, model=model)
    else:
        print('Invalid segment type')
        exit()

    res = segmentation_model.segment([stems(doc) for doc in docs])

    print('===結果===')
    print(res)
    save_path = './result/segmentation/' + segmentation_type + '/' + model_type + '/' + doc_type + '/' + model_type + '_doc_num_' + doc_num +'_window_size_' + str(segmentation_model.window_size) + '_' + str(datetime.date.today())

    fig = plt.figure()
    segmentation_model.sim_arr.plot(title=model_type)
    # segmentation_model.sim_arr.plot(title=model_type, yticks=[0, 0.5, 1.0])
    plt.savefig(save_path + '.png')
    plt.close('all')

    save_path = './result/segmentation/' + segmentation_type + '/' + model_type + '/' + doc_type  + '/interview_' + model_type + '_doc_num_' + doc_num +'_window_size_' + str(segmentation_model.window_size) + '_' + str(datetime.date.today())
    with open(save_path + '.txt', 'w') as f:
        for i in range(len(docs)):
            print(label[i] + '　' + docs[i].replace('\n','。'), file=f)
            print('', file=f)
            if str(i + 0.5) in res.index.values:
                print("___________\n", file=f)

    print('===評価===')
    if eval:
        path_for_eval = './data/interview-text_eval_' + doc_num + '.txt'
        sentence = False
        if doc_type == 'sentence':
            sentence = True
        data_for_eval = utils.load_data_for_eval(path_for_eval, sentence=sentence)
        label_for_eval = [item[0] for item in data_for_eval.items() if '____' in item[1][0]]

        print('予測:', res.index.values)
        print('正解:', label_for_eval)

        # 精度、再現率
        count = 0
        for i in res.index.values:
            i = float(i)
            # 前後許容
            if i in label_for_eval or i + 1 in label_for_eval or i - 1 in label_for_eval:
                count += 1
        p = count / len(res.index.values)
        r = count / len(label_for_eval)
        f_score = 2 * 1 / (1 / r + 1 / p)
        print("Precision", p)
        print("Recall", r)
        print("F score", f_score)

        save_path = './result/segmentation/' + segmentation_type + '/' + model_type + '/' + doc_type + '/evaluation_' + model_type + '_doc_num_' + doc_num +'_window_size_' + str(segmentation_model.window_size) + '_' + str(datetime.date.today())

        with open(save_path + '.txt', 'w') as f:
            print('予測:', res.index.values, file=f)
            print('正解:', label_for_eval, file=f)
            print("Precision", p, file=f)
            print("Recall", r, file=f)
            print("F score", f_score, file=f)
