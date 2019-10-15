from lib import scraping
from lib.tfidf import TfidfModel
from lib.doc2vec import Doc2Vec
from lib.word2vec import Word2Vec
from lib.utils import stems
from lib import utils
import sys


if __name__ == '__main__':
    update = False

    args = sys.argv
    if 2 <= len(args):
        if not(args[1] == 'tfidf' or args[1] == 'doc2vec' or args[1] == 'word2vec'):
            print('Argument is invalid')
            exit()
        if args[-1] == 'update':
            update = True
    else:
        print('Arguments are too sort')
        exit()

    model_type = args[1]

    # docs: インタビュー全体
    print('Load data')
    # モデルを訓練する
    data = utils.to_sentence(scraping.scraping(10))
    docs = [row[1] for row in data]

    # max_characters: XX文字以上の単文は要約対象外
    # docs = utils.polish_docs(docs, max_characters=1000)
    docs_for_train = [stems(doc) for doc in docs]
    """
    以下のようなデータを作っています
    edocs_for_train = [
    ['出身は', 'どこ', 'ですか' ...
    ['好き', 'な', '食べもの', ...
    ...
    ]
    """
    print(data[:3])
    print(docs[:1])
    print(docs_for_train[:1])
    print('Done')

    if model_type == 'tfidf':
        # TFIDFモデル生成
        # GensimのTFIDFモデルを用いた文のベクトル化
        print('===TFIDFモデル生成===')
        print('Train tfidf model')
        tfidf = TfidfModel(no_below=10, no_above=0.1, keep_n=100000)
        tfidf.train(docs_for_train)
        print('Done')

    elif model_type == 'doc2vec':
        print('===Doc2Vec===')
        label = [row[0] for row in data]
        label_docs = list(range(len(docs)))
        doc2vec = Doc2Vec(alpha=0.025, min_count=10, vector_size=300, epochs=50, workers=4)
        if update:
            print('Update doc2vec model')
            label_docs = [False for x in range(len(docs))]
            doc2vec.load_model('./model/doc2vec/doc2vec_wiki.model')
            doc2vec.update(docs_for_train, label_docs)
        else:
            print('Train doc2vec model')
            doc2vec.train(docs_for_train, label_docs)
        print('Done')

    elif model_type == 'word2vec':
        word2vec = Word2Vec(alpha=0.025, min_count=10, vector_size=200, epochs=50, workers=4)
        if update:
            print('Update word2vec model')
            word2vec.load_model('./model/word2vec/word2vec_wiki.model')
            word2vec.update(docs_for_train)
        else:
            print('Train word2vec model')
            word2vec.train(docs_for_train)
        print('Done')

    else:
        print('Invalid model type')
        exit()
