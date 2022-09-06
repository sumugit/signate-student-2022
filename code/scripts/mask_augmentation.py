import os
import copy
import setup
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from configuration import Config
from transformers import AutoTokenizer, AutoModelForMaskedLM


class BERTAugmenter():
    def __init__(self, model_name, num_sample_tokens):
        self.model_name = model_name
        self.num_sample_tokens = num_sample_tokens
    
    
    # 文章中の最初の [MASK] をスコアの上位の token に置き換える
    # 上位何位まで使うかは, num_topk で指定
    # 出力は穴埋めされた文章のリストと, 置き換えられた token のスコアのリスト
    def predcit_mask_topk(self, text, tokenizer, bert_mlm, num_topk):
        # 文章を符号化し, BERT で分類スコアを得る
        input_ids = tokenizer.encode(text, return_tensors='pt')
        input_ids = input_ids.cuda()
        with torch.no_grad():
            output = bert_mlm(input_ids=input_ids)
        scores = output.logits
        
        # スコアが上位の token とスコアを求める
        mask_position = input_ids[0].tolist().index(128000)  # [MASK] の token id (128000) の最初の位置
        topk = scores[0, mask_position].topk(num_topk)
        ids_topk = topk.indices  # token の id
        tokens_topk = tokenizer.convert_ids_to_tokens(ids_topk)  # token
        scores_topk = topk.values.cpu().numpy()  # score
        
        # 文章中の [MASK] を上で求めた token で置き換える
        text_topk = []  # 穴埋めされた文書を追加する
        for token in tokens_topk:
            token = token.replace("▁", "")  # 最初のアンダーバーを消す
            text_topk.append(text.replace("[MASK]", token, 1))  # 最大置換回数は 1
        
        return text_topk, scores_topk
    
    # beam search で文章の穴埋めを行う
    def beam_search(self, text, tokenizer, bert_mlm, num_topk):
        num_mask = text.count("[MASK]")
        text_topk = [text]
        scores_topk = np.array([0])
        for _ in range(num_mask):
            # 現在得られているそれぞれの文章に対して,
            # 「最初の」 [MASK] をスコアが上位のトークンで穴埋めする
            text_candidates = []  # それぞれの文章を穴埋めした結果を追加する
            score_candidates = []  # 穴埋めに使ったトークンのスコアを追加する
            for text_mask, score in zip(text_topk, scores_topk):
                text_topk_inner, scores_topk_inner = self.predcit_mask_topk(
                    text_mask, tokenizer, bert_mlm, num_topk
                )
                text_candidates.extend(text_topk_inner)
                score_candidates.append(score + scores_topk_inner)
        
            # 穴埋めにより生成された文章の中から合計スコアの高いものを選ぶ
            score_candidates = np.hstack(score_candidates)
            idx_list = score_candidates.argsort()[::-1][:num_topk] # 必要に応じて変える
            text_topk = [text_candidates[idx] for idx in idx_list]
            scores_topk = score_candidates[idx_list]

        return text_topk

    def _generate(self, text, num_replace_token, num_sample_tokens, bert_mlm, tokenizer):
        text_replaced = text
        # [MASK] する token の数だけ繰り返す
        idx_remove = []
        for _ in range(num_replace_token):
            replace_idx = -1
            words_list = text_replaced.split()
            # 1 文字は除く
            while (replace_idx < 0) or (len(words_list[replace_idx]) == 1) or (replace_idx in idx_remove):
                # 置き換える単語を抽出
                replace_idx = random.randrange(len(words_list))
            # mask token に置き換え
            text_replaced = text_replaced.replace(words_list[replace_idx], "[MASK]", 1)
            idx_remove.append(replace_idx)
        
        aug_texts = []
        
        if num_replace_token > 0:
            # [MASK] トークンを埋める
            text_topk = self.beam_search(text_replaced, tokenizer, bert_mlm, num_sample_tokens)
            for aug_text in text_topk:
                # Special token を消す
                _text = aug_text.replace(
                    "[CLS] ", "").replace(" [SEP]", "")
                # 変更前の文と同じ場合何もしない
                if _text == text:
                    continue
                aug_texts.append(_text)
        
        return aug_texts


    # 各文書を水増しする
    def generate(self, texts, targets, lengths, num_replace_tokens, bert_mlm, tokenizer):
        aug_texts = copy.deepcopy(texts)
        aug_targets = copy.deepcopy(targets)
        aug_lengths = copy.deepcopy(lengths)
        # 1 文書ずつ処理
        for idx, (text, target, length) in tqdm(enumerate(zip(texts, targets, lengths)), total=len(texts)):
            _text = self._generate(text, num_replace_tokens[idx], self.num_sample_tokens, bert_mlm, tokenizer)
            # リストの連結
            aug_texts += _text
            # その他の変数 (水増しした文書分だけ複製)
            aug_targets += [target for _ in range(len(_text))]
            aug_lengths += [length for _ in range(len(_text))]
        return aug_texts, aug_targets, aug_lengths


# 構成のセットアップ
cfg = setup.setup(Config)
# データ読込
dataset_train = pd.read_csv(os.path.join(cfg.INPUT, "train_preprocessed.csv"))
model = AutoModelForMaskedLM.from_pretrained(cfg.MODEL_PATH2)
tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_PATH)
model.cuda()

transformer_augmenter = BERTAugmenter( 
    model_name=cfg.MODEL_PATH,
    num_sample_tokens=cfg.num_sample_tokens,  # 生成する文書数
    # num_replace_tokens=cfg.num_replace_tokens
)

# pretrain
texts = dataset_train["description"].iloc[0:1].tolist()
targets = dataset_train["jobflag"].iloc[0:1].tolist()
lengths = dataset_train["len"].iloc[0:1].tolist()
num_replace_tokens = [int(val * cfg.mask_rate) for val in lengths]

# mask augmentation
aug_texts, aug_targets, aug_lengths = transformer_augmenter.generate(texts, targets, lengths, num_replace_tokens, model, tokenizer)

aug_df = pd.DataFrame({
    "description": aug_texts,
    "len": aug_lengths,
    "jobflag": aug_targets
})
aug_df.index.name = "id"
aug_df.to_csv(os.path.join(cfg.INPUT, "train_mask_augmented.csv"), index=True)
