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
import copy
sns.set()

import datetime
import sys



def validate_args(args):
    eval = False
    if 2 <= len(args):
        if not(args[1] == 'tfidf' or args[1] == 'doc2vec' or args[1] == 'word2vec'):
            print('Argument is invalid')
            exit()

        if not(args[2] == 'sentence' or args[2] == 'utterance'):
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

    return args[1], args[2], args[3], eval


def load_model(model_type, segmentation_type):
    if model_type == 'tfidf':
        # TFIDFモデル
        model = TfidfModel(no_below=1, no_above=1.0, keep_n=100000)
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

    if segmentation_type == 'text_tiling':
        segmentation_model = TextTiling(window_size=window_size, p_limit=0.1, a=0.5, model=model)
    elif segmentation_type == 'lcseg':
        segmentation_model = LexicalCohesionSegmentation(window_size=window_size, hiatus=11, p_limit=0.1, a=0.5, model=model)
    else:
        print('Invalid segment type')
        exit()

    return model, segmentation_model


def evaluation(res, segmentation_model, segmentation_type, model_type, doc_type, doc_num):
    count = 0
    label_for_eval = []
    path_for_eval = './data/eval/interview-text_' + doc_type + '_' + doc_num + '.txt'
    sentence = False
    if doc_type == 'sentence':
        sentence = True
    data_for_eval = utils.load_data_for_eval(path_for_eval, sentence=sentence)
    label_for_eval = [item[0] for item in data_for_eval.items() if '____' in item[1][0]]

    print('予測:', res.index.values)
    print('正解:', label_for_eval)

    # 精度、再現率
    plus_label_for_eval = [val + 1 for val in label_for_eval]
    minas_label_for_eval = [val - 1 for val in label_for_eval]
    approx_label_for_eval = copy.deepcopy(label_for_eval)
    approx_label_for_eval.extend(plus_label_for_eval)
    approx_label_for_eval.extend(minas_label_for_eval)
    for i in res.index.values:
        i = float(i)
        # 前後許容
        if i in approx_label_for_eval or i + 1 in approx_label_for_eval or i - 1 in approx_label_for_eval:
            count += 1

    p = count / len(res.index.values)
    r = count / len(label_for_eval)
    f_score = 2 * 1 / (1 / r + 1 / p)
    print("Precision:", p)
    print("Recall:", r)
    print("F score:", f_score)

    save_path = './result/segmentation/' + segmentation_type + '/' + model_type + '/' + doc_type + '/evaluation/' + 'doc_num_' + doc_num + '_' + model_type + '_window_size_' + str(segmentation_model.window_size) + '_' + str(datetime.date.today())

    with open(save_path + '.txt', 'w') as f:
        print('予測:', res.index.values, file=f)
        print('正解:', label_for_eval, file=f)
        print("Precision:", p, file=f)
        print("Recall:", r, file=f)
        print("F score:", f_score, file=f)

    return count, label_for_eval, f_score


def main_segmentation(doc_num, window_size, model_type, doc_type, segmentation_type, eval=False):
    # === Load doc ===
    print('')
    print('Interview:', doc_num)
    print('Load data')
    path = './data/interview/interview-text_01-26_' + doc_num + '.txt'

    data = utils.load_data(path)
    if doc_type == 'sentence':
        data = utils.to_sentence(data)

    docs = [row[1] for row in data]
    label = [row[0] for row in data]
    print(data[:5])
    print('Done')

    # === Model ===
    print('Model:', model_type)
    print('Segmentation type:', segmentation_type)
    model, segmentation_model = load_model(model_type, segmentation_type)

    # === Result ===
    print('===結果===')
    res = segmentation_model.segment([stems(doc) for doc in docs])
    # print(res)

    # 画像
    save_path = './result/segmentation/' + segmentation_type + '/' + model_type + '/' + doc_type + '/img/' + 'doc_num_' + doc_num + '_' + model_type + '_window_size_' + str(segmentation_model.window_size) + '_' + str(datetime.date.today())

    fig = plt.figure()
    plt.ylim([0, 1])
    segmentation_model.sim_arr.plot(title='Cosine similarity')
    plt.savefig(save_path + '.png')
    plt.close('all')

    # セグメント
    # save_path = './result/segmentation/' + segmentation_type + '/' + model_type + '/' + doc_type + '/interview_text/' + 'doc_num_' + doc_num + '_' + model_type + '_window_size_' + str(segmentation_model.window_size) + '_' + str(datetime.date.today())
    # For lda
    save_path = './data/segmentation/' + doc_type + '/' + 'interview-text_' + doc_num
    with open(save_path + '.txt', 'w') as f:
        for i in range(len(docs)):
            print(label[i] + '　' + docs[i].replace('\n','。'), file=f)
            print('', file=f)
            if str(i + 0.5) in res.index.values:
                print("___________\n", file=f)

    # === Evaluation ===
    count, f_score = 0, 0
    label_for_eval = []
    if eval:
        print('===評価===')
        count, label_for_eval, f_score = evaluation(res, segmentation_model, segmentation_type, model_type, doc_type, doc_num)

    return count, res.index.values, label_for_eval, f_score


if __name__ == '__main__':
    # ハイパーパラメータ
    window_size = 5
    doc_num = '26'

    # 引数
    model_type, doc_type, segmentation_type, eval = validate_args(sys.argv)

    count = []
    prediction = []
    ans = []
    f_score_arr = pd.Series()
    for num in range(int(doc_num)):
        num += 1
        if num < 10:
            num = '0' + str(num)
        else:
            num = str(num)

        tmp_count, tmp_prediction, tmp_ans, tmp_f_score = main_segmentation(num, window_size, model_type, doc_type, segmentation_type, eval=eval)
        count.append(tmp_count)
        prediction.append(tmp_prediction)
        ans.append(tmp_ans)
        f_score_arr[num] = tmp_f_score

    if eval:
        sum_count= 0
        sum_prediction = 0
        sum_ans = 0
        # Total
        for i in range(len(count)):
            sum_count += count[i]
            sum_prediction += len(prediction[i])
            sum_ans += len(ans[i])

        p = sum_count / sum_prediction
        r = sum_count / sum_ans
        f_score = 2 * 1 / (1 / r + 1 / p)
        print("--------------------")
        print("Total")
        print("Precision:", p)
        print("Recall:", r)
        print("F score:", f_score)

        save_path = './result/segmentation/' + segmentation_type + '/' + model_type + '/' + doc_type + '/doc_num_01_' + str(doc_num) + '_evaluation_' + model_type + '_window_size_' + str(window_size) + '_' + str(datetime.date.today())
        with open(save_path + '.txt', 'w') as f:
            print("Precision:", p, file=f)
            print("Recall:", r, file=f)
            print("F score:", f_score, file=f)

        # 画像
        fig = plt.figure()
        plt.ylim([0, 1])
        plt.ylabel('F score')
        plt.xlabel('Interview')
        f_score_arr.plot()
        # segmentation_model.sim_arr.plot(title=model_type, yticks=[0, 0.5, 1.0])
        plt.show()
        plt.close('all')

