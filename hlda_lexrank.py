from lib.py3hlda.sampler import HierarchicalLDA
# 自作のデータ読み込み&前処理用ライブラリ
from lib.tfidf import TfidfModel
from lib.utils import stems
from lib.utils import stopwords
from lib import utils
import gensim
from pprint import pprint
import sys
import _pickle as cPickle
import gzip
import os
from lib.summarize import summarize


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

        node = path[level]
        if node.node_id not in res.keys():
            res[node.node_id] = {'node': node, 'docs': []}
        res[node.node_id]['docs'].append(doc)
    return res


def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = cPickle.load(f)
        return loaded_object


def load_data_for_segmentation(doc_num, *, ans=False):
    print('Interview:', doc_num)
    path = './data/segmentation/sentence/interview-text_' + doc_num + '.txt'
    if ans:
        path = './data/eval/interview-text_sentence_' + doc_num + '.txt'

    return utils.load_data_segment(path)


def lexrank(topic, level, node, original_docs, dir='./result/summary/hlda/'):
    n_words = 10
    with_weights = False
    docs = []
    for arr in original_docs:
        tmp_docs = []
        for speaker, remark in arr:
            tmp_docs.append(remark)
        docs.append('\n'.join(tmp_docs))

    tfidf = TfidfModel(no_below=0, no_above=1.0, keep_n=100000)
    docs_for_training = [stems(doc) for doc in docs]
    tfidf.train(docs_for_training)
    sent_vecs = tfidf.to_vector(docs_for_training)
    # 表示
    print('===要約===')
    # 要約
    indexes = summarize(docs, sent_vecs, sort_type='normal', sent_limit=5, threshold=0.1)
    docs_summary = [original_docs[i] for i in indexes]

    path = []
    node_parent = node.parent
    while node_parent is not None:
        path.append(node_parent.node_id)
        node_parent = node_parent.parent
    path.reverse()
    for node_id in path:
        dir += '/topic_' + str(node_id)
    if not(os.path.exists(dir)):
        os.makedirs(dir)
    with open(dir + '/topic_' + str(topic) + '.txt', 'w') as f:
        node_parent = node.parent
        msg = 'topic=%d level=%d (documents=%d): ' % (node_parent.node_id, node_parent.level, node_parent.customers)
        msg += node_parent.get_top_words(n_words, with_weights)
        print(msg, file=f)
        msg = '    topic=%d level=%d (documents=%d): ' % (node.node_id, node.level, node.customers)
        msg += node.get_top_words(n_words, with_weights)
        print(msg, file=f)
        for node_child in node.children:
            msg = '        topic=%d level=%d (documents=%d): ' % (node_child.node_id, node_child.level, node_child.customers)
            msg += node_child.get_top_words(n_words, with_weights)
            print(msg, file=f)
        print('-------------------------------', file=f)
        for i, docs in enumerate(docs_summary):
            print('', file=f)
            print(str(i+1) + ':', file=f)
            for speaker, remark in docs:
                print(speaker + '　' + remark, file=f)


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

    doc_num = 'all'
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

        original_docs = []
        for data in data_arr:
            tmp_docs = []
            for item in data.items():
                if '_____' in item[1][0]:
                    original_docs.append(tmp_docs)
                    tmp_docs = []
                else:
                    tmp_docs.append((item[1][0], item[1][1]))
            original_docs.append(tmp_docs)

        # セグメント単位でまとめる
        docs = []
        for arr in original_docs:
            tmp_docs = []
            for speaker, remark in arr:
                tmp_docs.append(remark)
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
    dictionary = gensim.corpora.Dictionary(docs_for_dict)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    # for sentence
    # Load
    # dictionary = gensim.corpora.Dictionary.load_from_text('./model/tfidf/dict_' + str(no_below) + '_' + str(int(no_above * 100)) + '_' + str(keep_n) + '.dict')
    corpus = list(map(dictionary.doc2bow, docs_for_training))

    n_samples = 500       # no of iterations for the sampler
    alpha = 1.0          # smoothing over level distributions
    gamma = 1.0           # CRP smoothing parameter; number of imaginary customers at next, as yet unused table
    eta = 0.5             # smoothing over topic-word distributions
    num_levels = 4        # the number of levels in the tree
    display_topics = 100   # the number of iterations between printing a brief summary of the topics so far
    n_words = 10          # the number of most probable words to print for each topic after model estimation
    with_weights = False  # whether to print the words with the weights

    eta_for_path = ''.join(str(eta).split('.'))
    alpha_for_path = str(alpha).split('.')[0]
    model_name = 'levels_' + str(num_levels) + '_alpha_' + alpha_for_path + '_eta_' + eta_for_path + '_interview_' + str(doc_num)

    hlda = load_zipped_pickle('./model/hlda/' + model_name + '.p')

    level = 2
    res = get_docs_topic(hlda, original_docs, level=level)
    print(res.keys())

    docs_for_tfidf = []

    dir = './result/summary/hlda/'
    if ans:
        dir += 'ans/'
    # for topic
    for key, ele in res.items():
        lexrank(key, level, ele['node'], ele['docs'], dir=dir+model_name)
