import random
import sys
from pprint import pprint

import gensim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from lib import utils
from lib.tfidf import TfidfModel
from lib.utils import stems, stopwords

random.seed(13)


def load_data_for_segmentation(doc_num):
    print('Interview:', doc_num)
    path = './data/segmentation/sentence/interview-text_' + doc_num + '.txt'
    # ans
    # path = './data/eval/interview-text_sentence_' + doc_num + '.txt'

    return utils.load_data_segment(path)


def load_data_per_interview(doc_num):
    print('Interview:', doc_num)
    path = './data/interview/interview-text_01-26_' + doc_num + '.txt'

    return utils.load_data(path)


if __name__ == '__main__':
    args = sys.argv
    if len(args) >= 2:
        if not (args[1] == 'sentence' or args[1] == 'segmentation' or args[1] == 'utterance' or args[1] == 'interview'):
            print('Argument is invalid')
            sys.exit()
        if not (args[2] == 'perplexity' or args[2] == 'coherence'):
            print('Argument is invalid')
            sys.exit()
    else:
        print('Arguments are too sort')
        sys.exit()

    doc_type = args[1]
    eval_type = args[2]

    doc_num = 'all'
    path = './data/interview/interview-text_01-26_' + doc_num + '.txt'

    if doc_type == 'sentence':
        data = utils.load_data(path)
        # to sentence
        data = utils.to_sentence(data)
        docs = [row[1] for row in data]

    if doc_type == 'utterance':
        data = utils.load_data(path)
        docs = [row[1] for row in data]

    if doc_type == 'interview':
        if doc_num == 'all':
            doc_num = '26'
        data_arr = []
        for num in range(int(doc_num)):
            num += 1
            if num < 10:
                num = '0' + str(num)
            else:
                num = str(num)
            data_arr.append(load_data_per_interview(num))

        docs = []
        for data in data_arr:
            tmp_docs = []
            for ele in data:
                tmp_docs.extend([ele[1]])
            docs.append('\n'.join(tmp_docs))

    elif doc_type == 'segmentation':
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

    if doc_num == 'all':
        doc_num = '26'
    doc_num = '01_' + doc_num

    # Params
    no_below = 3
    no_above = 0.8
    keep_n = 100000
    # Metrics for Topic Models
    start = 2
    limit = 51
    step = 1

    sw = stopwords()
    # data_set = [stems(doc, polish=True, sw=sw) for doc in docs]
    # docs_for_dict = data_set

    print('===コーパス生成===')
    if eval_type == 'perplexity':
        # Test set
        print(docs[:3])
        random.shuffle(docs)
        print(docs[:3])
        test_size = int(len(docs) * 0.25)
        docs_test = docs[:test_size]
        # docs_test = docs
        test_set = [stems(doc, polish=True, sw=sw) for doc in docs_test]
        # dict
        # data_for_test_dict = [stems(doc, polish=True, sw=sw) for doc in utils.to_sentence_docs(docs_test)]
        data_for_test_dict = test_set
        test_dict = gensim.corpora.Dictionary(data_for_test_dict)
        test_dict.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
        test_corpus = list(map(test_dict.doc2bow, test_set))

        # Train set
        docs_train = docs[test_size:]
        # docs_train = docs
        train_set = [stems(doc, polish=True, sw=sw) for doc in docs_train]
        # dict
        # data_for_train_dict = [stems(doc, polish=True, sw=sw) for doc in utils.to_sentence_docs(docs_train)]
        data_for_train_dict = train_set
        train_dict = gensim.corpora.Dictionary(data_for_train_dict)
        train_dict.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
        train_corpus = list(map(train_dict.doc2bow, train_set))

        perplexity_vals = []
    elif eval_type == 'coherence':
        # Train set
        docs_train = docs
        train_set = [stems(doc, polish=True, sw=sw) for doc in docs_train]
        # dict
        # data_for_train_dict = [stems(doc, polish=True, sw=sw) for doc in utils.to_sentence_docs(docs_train)]
        docs_for_train_dict = train_set
        train_dict = gensim.corpora.Dictionary(docs_for_train_dict)
        train_dict.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
        train_corpus = list(map(train_dict.doc2bow, train_set))

        # tfidf = TfidfModel(no_below=no_below, no_above=no_above, keep_n=keep_n)
        # tfidf.train(train_set)
        # train_dict = tfidf.dictionary
        # train_corpus = tfidf.corpus
        # train_corpus = tfidf.model[train_corpus]

        coherence_vals = []

    # for doc1
    # sentence corpus
    # data_for_dict = utils.load_data(path)
    # data_for_dict = utils.to_sentence(data_for_dict)
    # docs_for_dict = [row[1] for row in data_for_dict]
    # docs_for_dict = [stems(doc, polish=True, sw=sw) for doc in docs_for_dict]

    print(docs[-3:])

    for n_topic in tqdm(range(start, limit, step)):
        # LDAモデルの構築
        lda_model = gensim.models.ldamodel.LdaModel(corpus=train_corpus, id2word=train_dict, num_topics=n_topic, random_state=1, iterations=1000)
        # lda_model = gensim.models.ldamulticore.LdaMulticore(corpus=train_corpus, id2word=train_dict, num_topics=n_topic, ranom_state=1, iterations=1000, workers=-1)
        if eval_type == 'perplexity':
            perplexity_vals.append(np.exp2(-lda_model.log_perplexity(test_corpus)))
        elif eval_type == 'coherence':
            coherence_model_lda = gensim.models.CoherenceModel(model=lda_model, texts=train_set, dictionary=train_dict, coherence='u_mass')
            coherence_vals.append(coherence_model_lda.get_coherence())

    # evaluation
    x = range(start, limit, step)

    color = 'black'
    fig = plt.figure(figsize=(12, 5))
    plt.xlabel('Number of Topics')
    plt.xticks(x)
    # perplexity
    if eval_type == 'perplexity':
        plt.ylabel('Perplexity')
        plt.plot(x, perplexity_vals, 'o-', color=color)
    # coherence
    elif eval_type == 'coherence':
        plt.ylabel('Coherence')
        plt.plot(x, coherence_vals, 'o-', color=color)
    plt.show()
