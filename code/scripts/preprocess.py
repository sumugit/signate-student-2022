import unicodedata
import re
import os
import pandas as pd
from bs4 import BeautifulSoup
import nltk
import warnings
from nltk.corpus import wordnet
warnings.filterwarnings("ignore")

PATH = "/home/sumiya/signate/student2022/code/deberta_v3_large/Input"


df_train = pd.read_csv(os.path.join(PATH, "train_li_exchanged_augmentation.csv"))
df_test = pd.read_csv(os.path.join(PATH, "test.csv"))
# train, test の結合
df_train["TrainFlag"] = True
df_test["TrainFlag"] = False
df = df_train.append(df_test)

# df = pd.read_csv(os.path.join(PATH, "li_records.csv"))

# 正規化
def normalize_text(text):
    text = re.sub('\r', '', text)     # 改行の除去
    text = re.sub('\n', '', text)     # 改行の除去
    text = re.sub('　', '', text)     # 全角空白の除去
    text = re.sub(r'\d+', '0', text)  # 数字文字の一律「0」化
    text = text.replace('e.g.', 'eg') # 特殊表記の変換
    text = text.replace('eg.', 'eg')  # 特殊表記の変換
    text = text.replace('ie.', 'ie')  # 特殊表記の変換
    text = text.replace('cf.', 'cf')  # 特殊表記の変換
    text = re.sub('\bex.', 'ex', text)# 特殊表記の変換
    text = text.replace('.', ' . ')   # ピリオドの前後に半角空白
    text = text.replace(',', ' , ')   # カンマの前後に半角空白
    text = text.replace('-', ' ')     # カンマの前後に半角空白
    # 記号の除去
    code_regex = re.compile(
        '[!"#$%&\\\\()*’+–/:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％]')
    text = code_regex.sub('', text)
    return text

# Unicode 正規化
def normalize_unicode(text, form='NFKC'):
    normalized_text = unicodedata.normalize(form, str(text))
    return normalized_text

# html タグの除去
def clean_html_tags(html_text):
    soup = BeautifulSoup(html_text, 'html.parser')
    cleaned_text = soup.get_text()
    cleaned_text = ''.join(cleaned_text.splitlines())
    return cleaned_text

# js コードの削除
def clean_html_and_js_tags(html_text):
    soup = BeautifulSoup(html_text, 'html.parser')
    [x.extract() for x in soup.findAll(['script', 'style'])]
    cleaned_text = soup.get_text()
    cleaned_text = ''.join(cleaned_text.splitlines())
    return cleaned_text

# url の削除
def clean_url(html_text):
    cleaned_text = re.sub(r'http\S+', '', html_text)
    return cleaned_text

# 全て小文字統一
def lower_text(text):
    return text.lower()

# Lemmatizing
def lemmatize_term(term, pos=None):
    if term == "has" or term == "as":
        return term
    elif pos is None:
        synsets = wordnet.synsets(term)
        if not synsets:
            return term
        pos = synsets[0].pos()
        if pos == wordnet.ADJ_SAT:
            pos = wordnet.ADJ
    return nltk.WordNetLemmatizer().lemmatize(term, pos=pos)

# 前処理まとめ
def preprocessing(text):
    text = normalize_unicode(text)
    text = clean_html_tags(text)
    text = clean_html_and_js_tags(text)
    text = clean_url(text)
    text = normalize_text(text)
    text = lower_text(text)
    text = ' '.join(lemmatize_term(e) for e in text.split())
    return str(text.strip())


# 前処理
df["description"] = df["description"].map(preprocessing)
df['len'] = df['description'].str.split().str.len()

# train
df_train = df[df["TrainFlag"] == True]
df_train= df_train.drop(["TrainFlag"], axis = 1)
df_train["jobflag"] = df_train["jobflag"].astype(int) - 1  # ラベルを 0 ~ 3 にする
# test
df_test = df[df["TrainFlag"] == False]
df_test= df_test.drop(["TrainFlag"], axis = 1)

# save
df_train.to_csv(os.path.join(PATH, "train_tag_preprocessed.csv"), index=False)
df_test.to_csv(os.path.join(PATH, "test_tag_preprocessed.csv"), index=False)


# df.to_csv(os.path.join(PATH, "li_records_preprocessed.csv"), index=False)