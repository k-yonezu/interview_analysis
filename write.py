from lib import utils
import csv

if __name__ == '__main__':
    update = False

    # docs: インタビュー全体
    print('Load data')
    # モデルを訓練する
    path = './data/interview/interview-text_01-26_all.txt'
    data = utils.to_sentence(utils.load_data(path))
    docs = [row[1] for row in data]

    with open('data/csv/docs.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        for i, doc in enumerate(docs):
            writer.writerow([i, doc])
