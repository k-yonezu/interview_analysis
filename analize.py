
# if __name__ == '__main__':

#     # docs: インタビュー全体
#     print('Load data')
#     # モデルを訓練する
#     path = './data/interview/interview-text_01-26_all.txt'
#     data = utils.to_sentence(utils.load_data(path))
#     docs = [row[1] for row in data]
#     print('Done')

#     print(docs[:1])
#     res = []
#     for i in range(len(docs)):
#         doc = docs[i]
#         if doc[-3:] == 'ですか' and 'そうですか' not in doc:
#             res.append(i - 0.5)
#     print(res[:100])

#     # max_characters: XX文字以上の単文は要約対象外
#     # docs = utils.polish_docs(docs, max_characters=1000)
#     # docs_for_train = [stems(doc) for doc in docs]
#     # """
#     # 以下のようなデータを作っています
#     # edocs_for_train = [
#     # ['出身は', 'どこ', 'ですか' ...
#     # ['好き', 'な', '食べもの', ...
#     # ...
#     # ]
#     # """
#     # print(data[:3])
#     # print(docs[:1])
#     # print(docs_for_train[:1])

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
        segmentation_model = LexicalCohesionSegmentation(window_size=window_size, hiatus=11, p_limit=0, a=0.5, model=model)
    else:
        print('Invalid segment type')
        exit()

    return model, segmentation_model


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
    print(segmentation_model.sim_arr)


if __name__ == '__main__':
    # ハイパーパラメータ
    window_size = 30
    doc_num = '2'

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

        main_segmentation(num, window_size, model_type, doc_type, segmentation_type, eval=eval)

