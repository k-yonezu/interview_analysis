from lib.py3hlda.sampler import HierarchicalLDA
# 自作のデータ読み込み&前処理用ライブラリ
from lib.tfidf import TfidfModel
from lib.utils import stems
from lib.utils import stopwords
from lib import utils
import gensim
from pprint import pprint
import datetime
import sys
import re
import pyLDAvis
import pyLDAvis.gensim
import _pickle as cPickle
import gzip


colour_map = {
    0: 'blue',
    1: 'red',
    2: 'green'
}

def show_doc(d=0):
    
    node = hlda.document_leaves[d]
    path = []
    while node is not None:
        path.append(node)
        node = node.parent
    path.reverse()   
    
    n_words = 10
    with_weights = False    
    for n in range(len(path)):
        node = path[n]
        colour = colour_map[n] 
        msg = 'Level %d Topic %d: ' % (node.level, node.node_id)
        msg += node.get_top_words(n_words, with_weights)
        output = '<h%d><span style="color:%s">%s</span></h3>' % (n+1, colour, msg)
        display(HTML(output))
        
    display(HTML('<hr/><h5>Processed Document</h5>'))

    doc = corpus[d]
    output = ''
    for n in range(len(doc)):
        w = doc[n]
        l = hlda.levels[d][n]
        colour = colour_map[l]
        output += '<span style="color:%s">%s</span> ' % (colour, w)
    display(HTML(output))

def get_docs_topic(hlda, docs, level=1):
    res = {}
    for i in range(len(docs)):
        doc = docs[i]
        node = hlda.document_leaves[i]
        path = []
        while node is not None:
            path.append(node)
            node = node.parent
        path.reverse()

        n_words = 15
        with_weights = False
        node = path[level]
        if node.node_id not in res.keys():
            res[node.node_id] = []
        res[node.node_id].append(doc)
    return res

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = cPickle.load(f)
        return loaded_object


def load_data_for_segmentation(doc_num, *, ans=False):
    print('Interview:',  doc_num)
    path = './data/segmentation/sentence/interview-text_' + doc_num + '.txt'
    # path = './data/segmentation/sentence/tmp_interview-text_' + doc_num + '.txt'
    # path = './data/segmentation/utterance/interview-text_' + doc_num + '.txt'
    if ans:
        path = './data/eval/interview-text_sentence_' + doc_num + '.txt'

    return utils.load_data_segment(path)


if __name__ == '__main__':
    args = sys.argv
    if 2 <= len(args):
        if not(args[1] == 'sentence' or args[1] == 'segmentation' or args[1] == 'utterance' or args[1] == 'segmentation/ans'):
            print('Argument is invalid')
            exit()
    else:
        print('Arguments are too sort')
        exit()

    doc_type = args[1]

    doc_num = '26'
    ans = False

    if doc_type == 'segmentation' or doc_type == 'segmentation/ans':
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

        # セグメント単位でまとめる
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

    path = './data/interview/interview-text_01-26_' + doc_num + '.txt'
    if doc_num == 'all':
        doc_num = '26'
    doc_num = '01_' + doc_num

    # Params
    no_below = 3
    no_above = 0.5
    keep_n = 100000
    sw = stopwords()
    docs_for_training = [stems(doc, polish=True, sw=sw) for doc in docs]
    docs_for_dict = docs_for_training

    # for doc1
    # sentence corpus
    # data_for_dict = utils.load_data(path)
    # data_for_dict = utils.to_sentence(data_for_dict)
    # docs_for_dict = [row[1] for row in data_for_dict]
    # docs_for_dict = [stems(doc, polish=True, sw=sw) for doc in docs_for_dict]

    # print('===コーパス生成===')
    # dictionary = gensim.corpora.Dictionary(docs_for_dict)
    # dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    # for sentence
    # Load
    dictionary = gensim.corpora.Dictionary.load_from_text('./model/tfidf/dict_' + str(no_below) + '_' + str(int(no_above * 100)) + '_' + str(keep_n) + '.dict')
    corpus = list(map(dictionary.doc2bow, docs_for_training))

    n_samples = 500       # no of iterations for the sampler
    alpha = 1.0          # smoothing over level distributions
    gamma = 1.0           # CRP smoothing parameter; number of imaginary customers at next, as yet unused table
    eta = 0.5             # smoothing over topic-word distributions
    num_levels = 4        # the number of levels in the tree
    display_topics = 100   # the number of iterations between printing a brief summary of the topics so far
    n_words =  5          # the number of most probable words to print for each topic after model estimation
    with_weights = False  # whether to print the words with the weights

    eta_for_path = ''.join(str(eta).split('.'))

    hlda = load_zipped_pickle('./model/hlda/level_' + str(num_levels) + '_eta_' + eta_for_path + '_interview_' + str(doc_num) + '.p')

    res = get_docs_topic(hlda, docs, level=1)
    print(res.keys())
    docs_for_tfidf = []
    for i, v in res.items():
        docs_for_tfidf.append('\n'.join(v))
    print(len(docs_for_tfidf))

    # tfidf
    docs_for_tfidf = [stems(doc) for doc in docs_for_tfidf]
    tfidf = TfidfModel(no_below=0, no_above=1, keep_n=keep_n)
    tfidf.train(docs_for_tfidf)
    dictionary = tfidf.dictionary
    corpus = tfidf.corpus
    corpus = tfidf.model[corpus]
    tfidf.save_model(dir='./model/tfidf/hlda/')
