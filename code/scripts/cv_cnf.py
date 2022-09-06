import os
import setup
import prediction
import numpy as np
import pandas as pd
from glob import glob
import deberta_v3_large_train
from configuration import Config
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings("ignore")


# 構成のセットアップ
cfg = setup.setup(Config)

# データの読込
dataset_train = pd.read_csv(os.path.join(cfg.INPUT, "train_mask_augmented.csv"))
dataset_test = pd.read_csv(os.path.join(cfg.INPUT, "test_preprocessed.csv"))
submission = pd.read_csv(os.path.join(cfg.INPUT, "submit_sample.csv"), header=None)

# tokenizerの読み込み
cfg.tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_PATH)

# validationデータの設定
cfg.folds = setup.get_stratifiedkfold(dataset_train, cfg.target, cfg.num_fold, cfg.seed)

# BERTの推論
cfg.model_weights = [p for p in sorted(
    glob(os.path.join(cfg.EXP_MODEL, 'fold*.pth')))]
sub_pred = prediction.inferring(cfg, dataset_test)

