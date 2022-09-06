import os
import setup
import prediction
import numpy as np
import pandas as pd
from glob import glob
from configuration import Config
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings("ignore")


# 構成のセットアップ
cfg = setup.setup(Config)
cfg.tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_PATH)

# データの読込
dataset_test = pd.read_csv(os.path.join(cfg.INPUT, "test_preprocessed.csv"))
dataset_test["description"] = dataset_test["description"].astype(str)
submission = pd.read_csv(os.path.join(
    cfg.INPUT, "submit_sample.csv"), header=None)

sub_pred = np.zeros(shape=(dataset_test.shape[0], 4))

ensemble_list = ["OUT_EX005", "OUT_EX014", "OUT_EX022"]
for out in ensemble_list:
    expxxx = os.path.join(cfg.OUTPUT, out)
    expxxx_model = os.path.join(expxxx, "model")
    # BERTの推論
    cfg.model_weights = [p for p in sorted(
        glob(os.path.join(expxxx_model, 'fold*.pth')))]
    sub_pred += prediction.inferring(cfg, dataset_test)


submission[1] = np.argmax(sub_pred, axis=1)
# クラスを 1 スタートに修正
submission[1] = submission[1].astype(int) + 1

# 提出用ファイル
submission.to_csv(os.path.join(cfg.FINAL, "submission_5_14_22.csv"),
                index=False, header=False)
