import re
import urllib.parse as parser
import urllib.request as request
from bs4 import BeautifulSoup
import sys


def get_docs_num(num):
    docs_num = []
    for i in range(num):
        param = i + 1
        link = "" + str(param)
        print(link)
        with request.urlopen(link) as response:
            html = response.read().decode("utf-8")
            soup = BeautifulSoup(html, "lxml")
            items = soup.find_all('a', class_='article_item')
            for e in items:
                docs_num.append(e.get('href').split('/').pop())

    return docs_num


def get_doc(doc_num):
    link = ""
    with request.urlopen(link + parser.quote_plus(str(doc_num))) as response:
        # BMP外を''に置換するマップ
        non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), '\n')
        html = response.read().decode("utf-8")
        soup = BeautifulSoup(html, "lxml")

        # title
        h1 = soup.find_all('h1')[0]
        title = str(doc_num) + ', ' + h1.text.translate(non_bmp_map).strip()

        # 本文抜き出し
        if not(soup.find_all(class_='article_parent')):
            return False, False

        body = soup.find_all(class_='article_parent')
        doc = body[0].text.translate(non_bmp_map)
        doc = re.sub("!", "\n", doc)
        doc = re.sub("！", "\n", doc)
        doc = re.sub("\?", "\n", doc)
        doc = re.sub("\u2003", " ", doc)
        # doc = re.sub(r'[︰-＠]', " ", doc)  # 全角記号
        # doc = re.sub(r'[-/:-@\[-`\{-~]', " ", doc)  # 半角記号
        doc = doc.replace("、", " ")
        doc = doc.replace("。", "\n")
        doc = doc.replace("\r", "\n")

    return title, doc.strip()


def scraping(num):
    docs = []
    docs_num = get_docs_num(num)
    # docs_num = ['4004', '4296']
    for doc_num in docs_num:
        print(doc_num)
        title, doc = get_doc(doc_num)
        if not(title):
            print("skip")
            continue
        docs.append((title, doc))

    return docs
