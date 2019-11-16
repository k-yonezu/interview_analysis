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


def load_data_for_segmentation(doc_num):
    print('Interview:',  doc_num)
    path = './data/eval/interview-text_sentence_' + doc_num + '.txt'

    return load_data(path)


def load_data(path):
    data = {}
    with open(path) as f:
        lines = f.readlines()
        i = 0
        for line in lines:
            # 実際の会話だけ抽出
            line_arr = line.split('　')
            doc = ''.join(line_arr[1:]).strip()

            if doc == '' and line_arr[0] == '':
                continue
            if doc == '' and line_arr[0] == '\n':
                continue

            # 発話単位
            if '_____' in line_arr[0] and i != 0:
                i -= 0.5
                data[i] = (line_arr[0], doc)
                i -= 0.5
            else:
                data[i] = (line_arr[0], doc)
            i += 1

    return data


def join_data(data):
    res = {}
    seg_flag = 0
    i = 0
    tmp = ''
    for index, ele in data.items():
        label = ele[0]
        doc = ele[1]
        if label != '__________\n' and doc == '':
            continue
        if label == '__________\n':
            if data[index - 0.5][0] == data[index + 0.5][0]:
                seg_flag = 1
            else:
                res[i - 0.5] = ('__________\n', '')
                seg_flag = 0
        else:
            if index + 1 not in data:
                res[i] = ele
                break
            if label == data[index + 1][0]:
                tmp += doc
            else:
                if seg_flag:
                    res[i] = (label, tmp + doc)
                    res[i + 0.5] = ('__________\n', '')
                else:
                    res[i] = ele

                tmp = ''
                seg_flag = 0
                i += 1

    return res


if __name__ == '__main__':
    # ハイパーパラメータ
    doc_num = '26'

    for num in range(int(doc_num)):
        num += 1
        if num < 10:
            num = '0' + str(num)
        else:
            num = str(num)

        data = load_data_for_segmentation(num)
        data = join_data(data)

        save_path = './data/eval/interview-text_utterance_' + num + '.txt'
        with open(save_path, 'w') as f:
            for index, ele in data.items():
                label = ele[0]
                doc = ele[1]
                if label == '__________\n':
                    print(label, file=f)
                else:
                    print(label + '　' + doc, file=f)
                print('', file=f)
