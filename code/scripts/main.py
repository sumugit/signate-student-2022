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
dataset_train = pd.read_csv(os.path.join(cfg.INPUT, "train.csv"))
dataset_test = pd.read_csv(os.path.join(cfg.INPUT, "test_preprocessed.csv"))
dataset_train["description"] = dataset_train["description"].astype(str)
dataset_test["description"] = dataset_test["description"].astype(str)
submission = pd.read_csv(os.path.join(cfg.INPUT, "submit_sample.csv"), header=None)

# tokenizerの読み込み
cfg.tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_PATH)


# validationデータの設定
cfg.folds = setup.get_stratifiedkfold(dataset_train, cfg.target, cfg.num_fold, cfg.seed)
cfg.folds.to_csv(os.path.join(cfg.EXP_PREDS, "folds.csv"), header=False)  # fold の index 保存

# deBERTa-v3-large の学習
score = deberta_v3_large_train.training(cfg, dataset_train, dataset_test)


# BERTの推論
cfg.model_weights = [p for p in sorted(
    glob(os.path.join(cfg.EXP_MODEL, 'fold*.pth')))]
sub_pred = prediction.inferring(cfg, dataset_test)

## 追記 ##
# dataset_test["pred"] = np.argmax(sub_pred, axis=1)
# dataset_test.to_csv(os.path.join(cfg.EXP_PREDS, 'confirm.csv')),
# grouped = dataset_test.groupby("id")["pred"].apply(lambda x: x.mode()).reset_index(name='pred')
## 終了 ##

submission[1] = np.argmax(sub_pred, axis=1)
# submission[1] = grouped["pred"]
# クラスを 1 スタートに修正
submission[1] = submission[1].astype(int) + 1

# 提出用ファイル
submission.to_csv(os.path.join(cfg.EXP_PREDS, 'submission.csv'),
        index=False, header=False)
