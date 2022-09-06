import os
import setup
import numpy as np
import pandas as pd
from configuration import Config
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")


# 構成のセットアップ
cfg = setup.setup(Config)
dataset_train = pd.read_csv(os.path.join(cfg.INPUT, "train.csv"))
oof_pred = np.load(os.path.join(cfg.EXP_PREDS, 'oof_pred.npy'))

score = f1_score(np.argmax(oof_pred, axis=1),
                dataset_train[cfg.target]-1, average='macro')

print(f'CV: {round(score, 5)}')
