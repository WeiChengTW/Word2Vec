# fasttext_train.py

from gensim.models import FastText
import jieba  # 若為中文語料，需安裝 jieba，英文可不需此步
import os


# 1. 讀取語料並分詞（以中文為例，英文可直接用 split）
def load_corpus(filepath):
    sentences = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 中文斷詞
            words = list(jieba.cut(line))
            # 英文可用：words = line.split()
            sentences.append(words)
    return sentences


# 2. 設定語料路徑
corpus_path = "wiki_seg.txt"  # 請將此檔案替換為你的語料檔案

# 3. 載入語料
sentences = load_corpus(corpus_path)

# 4. 訓練 fastText 模型
model = FastText(
    sentences=sentences,
    vector_size=100,  # 詞向量維度
    window=5,  # 上下文窗口
    min_count=5,  # 最小詞頻
    workers=4,  # 執行緒數
    sg=1,  # 1: skip-gram, 0: CBOW
    epochs=10,  # 訓練輪數
    min_n=3,  # 子詞最小長度
    max_n=6,  # 子詞最大長度
)

# 5. 儲存模型
model.save("fasttext_wiki.model")

# 6. 詞向量查詢範例
print(model.wv["維基百科"])  # 查詢詞向量
print(model.wv.most_similar("維基百科", topn=5))  # 查詢相似詞

# 7. 若需儲存為 word2vec 格式
model.wv.save_word2vec_format("fasttext_wiki.vec", binary=False)
