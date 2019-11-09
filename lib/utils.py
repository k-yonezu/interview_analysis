import MeCab
from gensim import models
import re
import urllib


def _split_to_words(text, *, to_stem=False, polish=False, sw=[]):
    """
    入力: 'すべて自分のほうへ'
    出力: tuple(['すべて', '自分', 'の', 'ほう', 'へ'])
    """
    tagger = MeCab.Tagger('mecabrc')  # 別のTaggerを使ってもいい
    mecab_result = tagger.parse(text)
    info_of_words = mecab_result.split('\n')
    words = []
    for info in info_of_words:
        # macabで分けると、文の最後に’’が、その手前に'EOS'が来る
        if info == 'EOS' or info == '':
            break
            # info => 'な\t助詞,終助詞,*,*,*,*,な,ナ,ナ'
        info_elems = info.split(',')

        if polish:
            if info_elems[0][-2:] != u"名詞" and info_elems[0][-3:] != u"形容詞" and info_elems[0][-2:] != u"動詞":
                continue
            if info_elems[0][-3:] == u"助動詞" or info_elems[1] == u"数" or info_elems[1] == u"非自立":
                continue
            # if info_elems[0][-2:] != u"名詞" or info_elems[1] != u"一般":
            # if info_elems[6] == u"ない":
            #     print(info_elems)
            #     continue
            if info_elems[6] in sw:
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
    slothlib_path = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
    slothlib_file = urllib.request.urlopen(url=slothlib_path)
    slothlib_stopwords = [line.decode("utf-8").strip() for line in slothlib_file]
    slothlib_stopwords = [ss for ss in slothlib_stopwords if not ss==u'']
    # print(slothlib_stopwords)

    # Merge and drop duplication
    stopwords += slothlib_stopwords
    stopwords = list(set(stopwords))

    return stopwords


def load_data_for_eval(path, sentence=False):
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


class LabeledListSentence(object):
    def __init__(self, words_list, labels):
        self.words_list = words_list
        self.labels = labels

    def __iter__(self):
        for i, words in enumerate(self.words_list):
            # yield models.doc2vec.LabeledSentence(words, ['%s' % self.labels[i]])
            label = ['%s' % self.labels[i]] if self.labels[i] else []
            yield models.doc2vec.LabeledSentence(words, label)
