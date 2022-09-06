import os
import random
import setup
import numpy as np
import pandas as pd
import warnings
from bs4 import BeautifulSoup
warnings.filterwarnings('ignore')
from configuration import Config


# 構成のセットアップ
cfg = setup.setup(Config)

# データの読込
train_raw = pd.read_csv(os.path.join(cfg.INPUT, "train_tag_preprocessed.csv"))
test_raw = pd.read_csv(os.path.join(cfg.INPUT, "test_tag_preprocessed.csv"))
dataset_train = pd.read_csv(os.path.join(cfg.INPUT, "train_preprocessed.csv"))
dataset_test = pd.read_csv(os.path.join(cfg.INPUT, "test_preprocessed.csv"))
dataset = pd.read_csv(os.path.join(cfg.INPUT, "li_records_preprocessed.csv"))


# processed されたデータを改行付けて text 保存
def mlm_preprocess_raw(dataset_train, dataset_test):
    data = pd.concat([dataset_train, dataset_test])
    # 1レコードずつ改行付けて保存.
    text = '\n'.join(data["description"].tolist())
    with open(os.path.join(cfg.INPUT, "train_text.txt"), 'w') as f:
        f.write(text)

# li タグ毎に区切って csv 保存
def mlm_li_raw(train_raw, test_raw):
    contents = []
    data = pd.concat([train_raw, test_raw])
    for idx, row in data.iterrows():
        soup = BeautifulSoup(row["description"], "html.parser")
        for text in soup.find_all("li"):
            contents.append((text.text.strip()))
    df = pd.DataFrame(contents, columns=["description"])
    df["description"] = df["description"].astype(str)
    df.to_csv(os.path.join(cfg.INPUT, "li_records.csv"), index=False)


def to_text(df):
    # 1レコードずつ改行付けて保存.
    df["description"] = df["description"].astype(str)
    text = '\n'.join(df["description"].tolist())
    with open(os.path.join(cfg.INPUT, "li_records.txt"), 'w') as f:
        f.write(str(text))


# train, test を li タグ毎に区切って csv 保存
def dataset_li_preprocess(train_raw, test_raw):
    contents = []
    for idx, row in train_raw.iterrows():
        soup = BeautifulSoup(row["description"], "html.parser")
        for text in soup.find_all("li"):
            contents.append((row["id"], text.text.strip(), row["jobflag"]))
    df_train = pd.DataFrame(contents, columns=["id", "description", "jobflag"])
    df_train["description"] = df_train["description"].astype(str)
    df_train.to_csv(os.path.join(cfg.INPUT, "train_li.csv"), index=False)

    contents = []
    for idx, row in test_raw.iterrows():
        soup = BeautifulSoup(row["description"], "html.parser")
        for text in soup.find_all("li"):
            contents.append((row["id"], text.text.strip()))
    df_test = pd.DataFrame(contents, columns=["id", "description"])
    df_test["description"] = df_test["description"].astype(str)
    df_test.to_csv(os.path.join(cfg.INPUT, "test_li.csv"), index=False)


mlm_preprocess_raw(dataset_train, dataset_test)
# to_text(dataset)
# dataset_li_preprocess(train_raw, test_raw)