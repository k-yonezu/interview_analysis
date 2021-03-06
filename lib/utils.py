import os
import re
import urllib

import MeCab
from gensim import models


def _split_to_words(text, *, to_stem=False, polish=False, sw=[]):
    """
    入力: 'すべて自分のほうへ'
    出力: tuple(['すべて', '自分', 'の', 'ほう', 'へ'])
    """
    tagger = MeCab.Tagger('mecabrc')  # 別のTaggerを使ってもいい
    mecab_result = tagger.parse(text)
    info_of_words = mecab_result.split('\n')
    words = []
    i = -1
    for info in info_of_words:
        i += 1
        # macabで分けると、文の最後に’’が、その手前に'EOS'が来る
        if info == 'EOS' or info == '':
            break
            # info => 'な\t助詞,終助詞,*,*,*,*,な,ナ,ナ'
        info_elems = info.split(',')

        # if info_elems[0][-2:] != u"名詞" and info_elems[0][-3:] != u"形容詞"  and info_elems[0][-2:] != u"動詞":
        #     continue
        # if info_elems[0][-3:] == u"助動詞" or info_elems[1] == u"非自立":
        #     continue

        if polish:
            if info_elems[0][-2:] != u"名詞":
            # if info_elems[0][-2:] != u"名詞" and info_elems[0][-3:] != u"形容詞"  and info_elems[0][-2:] != u"動詞":
            # if info_elems[0][-2:] != u"名詞" and info_elems[0][-3:] != u"形容詞":
                continue
            if info_elems[1] == u"形容動詞語幹" or info_elems[1] == u"副詞可能" or info_elems[1] == u"数" or info_elems[1] == u"非自立" or info_elems[1] == u"接尾":
                continue
            if info_elems[0][-3:] == u"助動詞":
                continue

            if info_elems[6] == u"ちょう":
                if i + 1 < len(info_of_words):
                    if info_of_words[i+1].split(',')[6] == u"ちん":
                        words.append(u"ちょうちん")
                        continue
            if info_elems[6] == u"まち":
                if i + 1 < len(info_of_words) and info_of_words[i+1] != 'EOS':
                    if info_of_words[i+1].split(',')[6] == u"づくり":
                        words.append(u"まちづくり")
                        continue
            if info_elems[6] == u"もの":
                if i + 1 < len(info_of_words) and info_of_words[i+1] != 'EOS':
                    if info_of_words[i+1].split(',')[6] == u"づくり":
                        words.append(u"ものづくり")
                        continue
            if info_elems[6] == u"商店":
                if i + 1 < len(info_of_words):
                    if info_of_words[i+1].split(',')[6] == u"街":
                        words.append(u"商店街")
                        continue
            if info_elems[6] == u"ハロ":
                if i + 1 < len(info_of_words):
                    if info_of_words[i+1].split(',')[6] == u"ウィーン":
                        words.append(u"ハロウィーン")
                        continue
            if info_elems[6] == u"高低":
                if i + 1 < len(info_of_words):
                    if info_of_words[i+1].split(',')[6] == u"差":
                        words.append(u"高低差")
                        continue
            if info_elems[6] == u"You":
                if i + 1 < len(info_of_words):
                    if info_of_words[i+1].split(',')[6] == u"Tube":
                        words.append(u"YouTube")
                continue
            # if info_elems[0][-2:] != u"名詞" or info_elems[1] != u"一般":
            # if info_elems[0][-2:] == u"動詞":
            #     print(info_elems)
            #     continue

            if info_elems[6] == u"街":
                if i - 1 >= 0:
                    if info_of_words[i-1].split(',')[6] == u"商店":
                        continue
                    else:
                        words.append(u"まち")
                        continue
            if info_elems[6] == u"町":
                words.append(u"まち")
                continue
            if info_elems[6] == u"まつり":
                words.append(u"祭り")
                continue
            if info_elems[6] in sw:
                continue
            if info_elems[6] == '*':
                if info_elems[0][:-3] in sw:
                    continue


        # 6番目に、無活用系の単語が入る。もし6番目が'*'だったら0番目を入れる
        if info_elems[6] == '*':
            # info_elems[0] => 'ヴァンロッサム\t名詞'
            words.append(info_elems[0][:-3])
            continue
        if to_stem:
            # 語幹に変換
            words.append(info_elems[6])
            continue
        # 語をそのまま
        words.append(info_elems[0][:-3])
    return words


def stems(text, *, polish=False, sw=[]):
    text = text.replace('＃＃＃＃','')
    text = text.replace('＊＊＊＊','')
    text = text.replace('＊','')
    text = text.replace('(',' ')
    text = text.replace(')',' ')
    text = text.replace('、',' ')
    text = text.replace('～',' ')
    text = text.replace('?',' ')
    text = text.replace('!',' ')
    text = text.replace('！',' ')
    text = text.replace('？',' ')
    text = text.replace('＠','')
    text = text.replace(':','')
    text = text.replace('::','')
    # if polish:
    #     text = re.sub(r'[0-9]+', '', text)

    stems = _split_to_words(text=text, to_stem=True, polish=polish, sw=sw)
    return stems


def stopwords():
    stopwords = []
    path = './data/stopwords/Japanese.txt'
    if os.path.exists(path):
        with open(path) as f:
            lines = f.readlines()
            for line in lines:
                stopwords.append(line.strip())
    else:
        print('Download sw from slothlib')
        path = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
        slothlib_file = urllib.request.urlopen(url=path)
        stopwords = [line.decode("utf-8").strip() for line in slothlib_file]

    stopwords = [ss for ss in stopwords if not ss==u'']
    stopwords = list(set(stopwords))

    return stopwords


def load_data_segment(path, sentence=False):
    data = {}
    with open(path) as f:
        lines = f.readlines()
        i = 0
        for line in lines:
            # 実際の会話だけ抽出
            line_arr = line.split('　')
            doc = ''.join(line_arr[1:]).strip()
            doc = doc.replace('。','\n')

            if doc == '' and '_____' not in line_arr[0]:
                continue


            # 分単位
            # docs_arr = doc.split('\n')
            # docs_arr.pop()
            # data.extend(docs_arr)
            # 話者単位
            if '_____' in line_arr[0] and i != 0:
                i -= 0.5
                data[i] = (line_arr[0], doc)
                i -= 0.5
            else:
                if sentence:
                    sent_arr = doc.split('\n')
                    for sent in sent_arr:
                        if sent.strip() != '':
                            data[i] = (line_arr[0], sent)
                            i += 1
                    continue
                else:
                    data[i] = (line_arr[0], doc)
            i += 1

    return data


def load_data(path):
    data = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            # 実際の会話だけ抽出
            line_arr = line.split('　')
            doc = ''.join(line_arr[1:]).strip()

            doc = doc.replace('。','\n')
            if doc == '':
                continue

            # 分単位
            # docs_arr = doc.split('\n')
            # docs_arr.pop()
            # data.extend(docs_arr)
            # 話者単位
            data.append((line_arr[0], doc))

    return data


def polish_docs(docs, max_characters=0):
    data = list(filter(lambda s: len(s) < max_characters, docs)) if max_characters else docs

    return docs


def to_sentence(data):
    res = []
    sent_arr = []
    for label, doc in data:
        tmp = []
        sent_arr = doc.split('\n')
        sent_arr.pop()
        tmp = [(label, sent) for sent in sent_arr if sent.strip() != '']
        res.extend(tmp)

    return res


def to_sentence_docs(docs):
    res = []
    sent_arr = []
    for doc in docs:
        tmp = []
        sent_arr = doc.split('\n')
        sent_arr.pop()
        tmp = [sent for sent in sent_arr if sent.strip() != '']
        res.extend(tmp)

    return res


class LabeledListSentence(object):
    def __init__(self, words_list, labels):
        self.words_list = words_list
        self.labels = labels

    def __iter__(self):
        for i, words in enumerate(self.words_list):
            # yield models.doc2vec.LabeledSentence(words, ['%s' % self.labels[i]])
            label = ['%s' % self.labels[i]] if self.labels[i] else []
            yield models.doc2vec.LabeledSentence(words, label)
