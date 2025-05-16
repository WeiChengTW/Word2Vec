# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] papermill={"duration": 0.032151, "end_time": "2021-01-06T19:08:06.349966", "exception": false, "start_time": "2021-01-06T19:08:06.317815", "status": "completed"}
# # Word2Vec-以 gensim 訓練中文詞向量
# ## 參考及引用資料來源
#
# - [1] [zake7749-使用 gensim 訓練中文詞向量](http://zake7749.github.io/2016/08/28/word2vec-with-gensim/)
# * [2] [gensim/corpora/wikicorpus](https://radimrehurek.com/gensim/corpora/wikicorpus.html)
# - [Word2Vec的簡易教學與參數調整指南](https://www.kaggle.com/jerrykuo7727/word2vec)
# * [zhconv](https://pypi.org/project/zhconv/)
# * [jieba](https://pypi.org/project/jieba/)
#
#
#
#

# %% papermill={"duration": 9.422107, "end_time": "2021-01-06T19:08:15.802634", "exception": false, "start_time": "2021-01-06T19:08:06.380527", "status": "completed"}
# %pip install memory_profiler

# %load_ext memory_profiler  # Uncomment this line if running in a Jupyter notebook

import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "zhconv"])

# %% [markdown] papermill={"duration": 0.030417, "end_time": "2021-01-06T19:08:15.864878", "exception": false, "start_time": "2021-01-06T19:08:15.834461", "status": "completed"}
# 確認相關 Packages

# %% _cell_guid="79c7e3d0-c299-4dcb-8224-4455121ee9b0" _uuid="d629ff2d2480ee46fbb7e2d37f6b5fab8052498a" papermill={"duration": 4.896957, "end_time": "2021-01-06T19:08:20.792497", "exception": false, "start_time": "2021-01-06T19:08:15.895540", "status": "completed"}
# %pip install scipy==1.12


import os

# Packages
import gensim
import jieba
import zhconv
from gensim.corpora import WikiCorpus
from datetime import datetime as dt
from typing import List


if not os.path.isfile("dict.txt.big"):
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "wget"])
    import wget

    wget.download(
        "https://github.com/fxsjy/jieba/raw/master/extra_dict/dict.txt.big",
        "dict.txt.big",
    )
jieba.set_dictionary("dict.txt.big")

print("gensim", gensim.__version__)
print("jieba", jieba.__version__)

# %% [markdown] papermill={"duration": 0.035697, "end_time": "2021-01-06T19:08:20.864355", "exception": false, "start_time": "2021-01-06T19:08:20.828658", "status": "completed"}
# # 準備中文訓練文本
#

# %% [markdown] papermill={"duration": 0.036708, "end_time": "2021-01-06T19:08:20.937019", "exception": false, "start_time": "2021-01-06T19:08:20.900311", "status": "completed"}
#
# ## 訓練文本來源: [維基百科資料庫](https://zh.wikipedia.org/wiki/Wikipedia:%E6%95%B0%E6%8D%AE%E5%BA%93%E4%B8%8B%E8%BD%BD)
# > 要訓練詞向量，第一步當然是取得資料集。由於 word2vec 是基於非監督式學習，**訓練集一定一定要越大越好，語料涵蓋的越全面，訓練出來的結果也會越漂亮**。[[1]](http://zake7749.github.io/2016/08/28/word2vec-with-gensim/)
#
# - [zhwiki-20210101-pages-articles.xml.bz2](https://dumps.wikimedia.org/zhwiki/20210101/zhwiki-20210101-pages-articles.xml.bz2) (1.9 GB)
#
# > ```
# wget "https://dumps.wikimedia.org/zhwiki/20210101/zhwiki-20210101-pages-articles.xml.bz2"
# ```
#
#

# %% [markdown] papermill={"duration": 0.035785, "end_time": "2021-01-06T19:08:21.010241", "exception": false, "start_time": "2021-01-06T19:08:20.974456", "status": "completed"}
# 目前已經使用另一份 Notebook ([維基百科中文語料庫 zhWiki_20210101](https://www.kaggle.com/bbqlp33/zhwiki-20210101)) 下載好中文維基百科語料，並可以直接引用

# %% papermill={"duration": 28.972913, "end_time": "2021-01-06T19:08:50.019822", "exception": false, "start_time": "2021-01-06T19:08:21.046909", "status": "completed"}
# ZhWiki = "/kaggle/input/zhwiki-20250401/zhwiki-20250401-pages-articles.xml.bz2"

# #!du -sh $ZhWiki
# #!md5sum $ZhWiki
# #!file $ZhWiki

# %%
import os

# 完整 wiki articles 下載位置
ZhWiki = "https://dumps.wikimedia.org/zhwiki/20250120/zhwiki-20250120-pages-articles-multistream.xml.bz2"


# Download
# 取得檔名
ZhWiki_filename = ZhWiki.split("/")[-1]

# 若檔案已存在則不下載
if not os.path.isfile(ZhWiki_filename):
    import subprocess

    subprocess.check_call(["wget", ZhWiki])

else:
    print(f"{ZhWiki_filename} 已存在，略過下載。")


# !du -sh $ZhWiki
# !md5sum $ZhWiki
# !file $ZhWiki

# %% [markdown] papermill={"duration": 0.036617, "end_time": "2021-01-06T19:08:50.100480", "exception": false, "start_time": "2021-01-06T19:08:50.063863", "status": "completed"}
# # 中文文本前處理
#
# 在正式訓練 `Word2Vec` 之前，其實涉及了文本的前處理，本篇的處理包括如下三點 (而實務上對應的不同使用情境，可能會有不同的前處理流程):
#
# * 簡轉繁: [zhconv](https://pypi.org/project/zhconv/)
# * 中文斷詞: [jieba](https://pypi.org/project/jieba/)
# * 停用詞

# %% [markdown] papermill={"duration": 0.043279, "end_time": "2021-01-06T19:08:50.181398", "exception": false, "start_time": "2021-01-06T19:08:50.138119", "status": "completed"}
# ## 簡繁轉換
#
# wiki 文本其實摻雜了簡體與繁體中文，比如「数学」與「數學」，這會被 word2vec 當成兩個不同的詞。[[1]](http://zake7749.github.io/2016/08/28/word2vec-with-gensim/)
# <br>
# 所以我們在斷詞前，需要加上簡繁轉換的手續
#
# ---
#
# 以下範例使用了較輕量的 Package [zhconv](https://pypi.org/project/zhconv/)，
# <br>
# 若需要更高的精準度，則可以參考  [OpenCC](https://github.com/BYVoid/OpenCC)

# %% papermill={"duration": 0.09025, "end_time": "2021-01-06T19:08:50.309026", "exception": false, "start_time": "2021-01-06T19:08:50.218776", "status": "completed"}
zhconv.convert("这原本是一段简体中文", "zh-tw")

# %% [markdown] papermill={"duration": 0.037294, "end_time": "2021-01-06T19:08:50.389213", "exception": false, "start_time": "2021-01-06T19:08:50.351919", "status": "completed"}
# ## 中文斷詞
# 使用 [jieba](https://pypi.org/project/jieba/) `jieba.cut` 來進行中文斷詞，
# <br>
# 並簡單介紹 jieba 的兩種分詞模式:
# * `cut_all=False` **精確模式**，試圖將句子最精確地切開，適合文本分析；
# * `cut_all=True` **全模式**，把句子中所有的可以成詞的詞語都掃描出來, 速度非常快，但是不能解決歧義；
#
# 而本篇文本訓練採用**精確模式** `cut_all=False`

# %% papermill={"duration": 2.168334, "end_time": "2021-01-06T19:08:52.594505", "exception": false, "start_time": "2021-01-06T19:08:50.426171", "status": "completed"}
seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
print("Full Mode: " + "/ ".join(seg_list))  # 全模式

seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))  # 精確模式

# %% papermill={"duration": 0.04889, "end_time": "2021-01-06T19:08:52.682675", "exception": false, "start_time": "2021-01-06T19:08:52.633785", "status": "completed"}
print(list(jieba.cut("中英夾雜的example，Word2Vec應該很interesting吧?")))

# %% [markdown] papermill={"duration": 0.038844, "end_time": "2021-01-06T19:08:52.762814", "exception": false, "start_time": "2021-01-06T19:08:52.723970", "status": "completed"}
# ## 引入停用詞表
#
# 停用詞就是像英文中的 **the,a,this**，中文的**你我他**，與其他詞相比顯得不怎麼重要，對文章主題也無關緊要的，
# <br>
# 是否要使用停用詞表，其實還是要看你的應用，也有可能保留這些停用詞更能達到你的目標。[[1]](http://zake7749.github.io/2016/08/28/word2vec-with-gensim/)
# <br>
#
#
# * [Is it compulsory to remove stop words with word2vec?](https://www.quora.com/Is-it-compulsory-to-remove-stop-words-with-word2vec)
# * [The Effect of Stopword Filtering prior to Word Embedding Training](https://stats.stackexchange.com/questions/201372/the-effect-of-stopword-filtering-prior-to-word-embedding-training)
#
# ---
#
# 以下範例還是示範引入停用詞表，而停用詞表網路上有各種各樣的資源
# <br>
# 剛好 `kaggle`，環境預設有裝 [spacy](https://pypi.org/project/spacy/)，
# <br>
# 就順道引用 `spacy` 提供的停用詞表吧 (實務上stopwords 應為另外準備好且檢視過的靜態文檔)

# %% papermill={"duration": 37.622117, "end_time": "2021-01-06T19:09:30.424436", "exception": false, "start_time": "2021-01-06T19:08:52.802319", "status": "completed"}
import spacy

# 下載語言模組
spacy.cli.download("zh_core_web_sm")  # 下載 spacy 中文模組
spacy.cli.download("en_core_web_sm")  # 下載 spacy 英文模組

nlp_zh = spacy.load("zh_core_web_sm")  # 載入 spacy 中文模組
nlp_en = spacy.load("en_core_web_sm")  # 載入 spacy 英文模組

# 印出前20個停用詞
print("--\n")
print(
    f"中文停用詞 Total={len(nlp_zh.Defaults.stop_words)}: {list(nlp_zh.Defaults.stop_words)[:20]} ..."
)
print("--")
print(
    f"英文停用詞 Total={len(nlp_en.Defaults.stop_words)}: {list(nlp_en.Defaults.stop_words)[:20]} ..."
)

# %% papermill={"duration": 0.063819, "end_time": "2021-01-06T19:09:30.529975", "exception": false, "start_time": "2021-01-06T19:09:30.466156", "status": "completed"}
STOPWORDS = (
    nlp_zh.Defaults.stop_words
    | nlp_en.Defaults.stop_words
    | set(["\n", "\r\n", "\t", " ", ""])
)
print(len(STOPWORDS))

# 將簡體停用詞轉成繁體，擴充停用詞表
for word in STOPWORDS.copy():
    STOPWORDS.add(zhconv.convert(word, "zh-tw"))

print(len(STOPWORDS))


# %% [markdown] papermill={"duration": 0.041166, "end_time": "2021-01-06T19:09:30.613556", "exception": false, "start_time": "2021-01-06T19:09:30.572390", "status": "completed"}
# # 讀取 wiki 語料庫，並且進行前處理和斷詞
#
# 維基百科 (`wiki.xml.bz2`)下載好後，先別急著解壓縮，因為這是一份 xml 文件，裏頭佈滿了各式各樣的標籤，我們得先想辦法送走這群不速之客，不過也別太擔心，`gensim` 早已看穿了一切，藉由調用 [wikiCorpus](https://radimrehurek.com/gensim/corpora/wikicorpus.html)，我們能很輕鬆的只取出文章的標題和內容。[[1]](http://zake7749.github.io/2016/08/28/word2vec-with-gensim/)
#
# ![image.png](attachment:image.png)
#
#  [[2]](https://radimrehurek.com/gensim/corpora/wikicorpus.html)
#
# ---
#
# Supported dump formats:
#
# - `<LANG>wiki-<YYYYMMDD>-pages-articles.xml.bz2`
#
# - `<LANG>wiki-latest-pages-articles.xml.bz2`
#
# The documents are extracted on-the-fly, so that the whole (massive) dump can stay compressed on disk.
#


# %% papermill={"duration": 0.051347, "end_time": "2021-01-06T19:09:30.706981", "exception": false, "start_time": "2021-01-06T19:09:30.655634", "status": "completed"}
def preprocess_and_tokenize(
    text: str, token_min_len: int = 1, token_max_len: int = 15, lower: bool = True
) -> List[str]:
    if lower:
        text = text.lower()
    text = zhconv.convert(text, "zh-tw")
    return [
        token
        for token in jieba.cut(text, cut_all=False)
        if token_min_len <= len(token) <= token_max_len and token not in STOPWORDS
    ]


# %% papermill={"duration": 0.052172, "end_time": "2021-01-06T19:09:30.800774", "exception": false, "start_time": "2021-01-06T19:09:30.748602", "status": "completed"}
print(
    preprocess_and_tokenize(
        "歐幾里得，西元前三世紀的古希臘數學家，現在被認為是幾何之父，此畫為拉斐爾"
    )
)
print(preprocess_and_tokenize("我来到北京清华大学"))
print(preprocess_and_tokenize("中英夾雜的example，Word2Vec應該很interesting吧?"))

# %% papermill={"duration": 8578.445907, "end_time": "2021-01-06T21:32:29.288882", "exception": false, "start_time": "2021-01-06T19:09:30.842975", "status": "completed"}
# %%time
# %%memit
ZhWiki_path = "zhwiki-20250120-pages-articles-multistream.xml.bz2"
print(f"Parsing {ZhWiki_path}...")
wiki_corpus = WikiCorpus(
    ZhWiki_path, tokenizer_func=preprocess_and_tokenize, token_min_len=1
)

# %% [markdown] papermill={"duration": 0.042937, "end_time": "2021-01-06T21:32:29.375141", "exception": false, "start_time": "2021-01-06T21:32:29.332204", "status": "completed"}
# 初始化`WikiCorpus`後，能藉由`get_texts()`可迭代每一篇文章，它所回傳的是一個`tokens list`，我以空白符將這些 `tokens` 串接起來，統一輸出到同一份文字檔裡。這邊要注意一件事，`get_texts()`受 `article_min_tokens` 參數的限制，只會回傳內容長度大於 **50** (default) 的文章。
#
# - **article_min_tokens** *(int, optional)* – Minimum tokens in article. Article will be ignored if number of tokens is less.

# %% [markdown] papermill={"duration": 0.042626, "end_time": "2021-01-06T21:32:29.460779", "exception": false, "start_time": "2021-01-06T21:32:29.418153", "status": "completed"}
# 秀出前 3 偏文章的前10 個 token

# %% papermill={"duration": 0.621833, "end_time": "2021-01-06T21:32:30.125442", "exception": false, "start_time": "2021-01-06T21:32:29.503609", "status": "completed"}
g = wiki_corpus.get_texts()
print(next(g)[:10])
print(next(g)[:10])
print(next(g)[:10])


# print(jieba.lcut("".join(next(g))[:50]))
# print(jieba.lcut("".join(next(g))[:50]))


# %% [markdown] papermill={"duration": 0.068246, "end_time": "2021-01-06T21:32:30.264774", "exception": false, "start_time": "2021-01-06T21:32:30.196528", "status": "completed"}
# ## 將處理完的語料集存下來，供後續使用

# %% papermill={"duration": 7761.972053, "end_time": "2021-01-06T23:41:52.305261", "exception": false, "start_time": "2021-01-06T21:32:30.333208", "status": "completed"}
WIKI_SEG_TXT = "wiki_seg.txt"

generator = wiki_corpus.get_texts()

with open(WIKI_SEG_TXT, "w", encoding="utf-8") as output:
    for texts_num, tokens in enumerate(generator):
        output.write(" ".join(tokens) + "\n")

        if (texts_num + 1) % 100000 == 0:
            print(f"[{str(dt.now()):.19}] 已寫入 {texts_num} 篇斷詞文章")

# %% [markdown] papermill={"duration": 0.045558, "end_time": "2021-01-06T23:41:52.396893", "exception": false, "start_time": "2021-01-06T23:41:52.351335", "status": "completed"}
# # 訓練 Word2Vec

# %% papermill={"duration": 3876.514063, "end_time": "2021-01-07T00:46:28.956705", "exception": false, "start_time": "2021-01-06T23:41:52.442642", "status": "completed"}
# %%time

from gensim.models import word2vec
import multiprocessing

max_cpu_counts = multiprocessing.cpu_count()
word_dim_size = 300  #  設定 word vector 維度
print(f"Use {max_cpu_counts} workers to train Word2Vec (dim={word_dim_size})")


# 讀取訓練語句
sentences = word2vec.LineSentence(WIKI_SEG_TXT)

# 訓練模型
model = word2vec.Word2Vec(sentences, vector_size=word_dim_size, workers=max_cpu_counts)

# 儲存模型
output_model = f"word2vec.zh.{word_dim_size}.model"
model.save(output_model)

# %% [markdown] papermill={"duration": 2.199742, "end_time": "2021-01-07T00:46:32.742916", "exception": false, "start_time": "2021-01-07T00:46:30.543174", "status": "completed"}
# 儲存的模型總共會產生三份檔案

# %% papermill={"duration": 1.281755, "end_time": "2021-01-07T00:46:35.212079", "exception": false, "start_time": "2021-01-07T00:46:33.930324", "status": "completed"}
# ! ls word2vec.zh*

# %% papermill={"duration": 1.12531, "end_time": "2021-01-07T00:46:36.615602", "exception": false, "start_time": "2021-01-07T00:46:35.490292", "status": "completed"}
# !du -sh word2vec.zh*

# %% [markdown] papermill={"duration": 0.059046, "end_time": "2021-01-07T00:46:45.152482", "exception": false, "start_time": "2021-01-07T00:46:45.093436", "status": "completed"}
#  # 查看模型以及詞向量實驗

# %% [markdown] papermill={"duration": 0.052219, "end_time": "2021-01-07T00:46:45.962639", "exception": false, "start_time": "2021-01-07T00:46:45.910420", "status": "completed"}
# 模型其實就是巨大的 Embedding Matrix
#

# %% papermill={"duration": 0.063148, "end_time": "2021-01-07T00:46:46.074295", "exception": false, "start_time": "2021-01-07T00:46:46.011147", "status": "completed"}
print(model.wv.vectors.shape)
model.wv.vectors

# %% [markdown] papermill={"duration": 0.04893, "end_time": "2021-01-07T00:46:46.180511", "exception": false, "start_time": "2021-01-07T00:46:46.131581", "status": "completed"}
# 收錄的詞彙

# %% papermill={"duration": 0.247284, "end_time": "2021-01-07T00:46:46.476801", "exception": false, "start_time": "2021-01-07T00:46:46.229517", "status": "completed"}
print(f"總共收錄了 {len(model.wv.vocab)} 個詞彙")

print("印出 20 個收錄詞彙:")
print(list(model.wv.vocab.keys())[:10])

# %% [markdown] papermill={"duration": 0.158624, "end_time": "2021-01-07T00:46:46.682787", "exception": false, "start_time": "2021-01-07T00:46:46.524163", "status": "completed"}
# 詞彙的向量

# %% papermill={"duration": 0.070434, "end_time": "2021-01-07T00:46:46.800247", "exception": false, "start_time": "2021-01-07T00:46:46.729813", "status": "completed"}
vec = model.wv["數學家"]
print(vec.shape)
vec

# %% [markdown] papermill={"duration": 0.05081, "end_time": "2021-01-07T00:46:46.897024", "exception": false, "start_time": "2021-01-07T00:46:46.846214", "status": "completed"}
# 沒見過的詞彙

# %% papermill={"duration": 0.055741, "end_time": "2021-01-07T00:46:47.002071", "exception": false, "start_time": "2021-01-07T00:46:46.946330", "status": "completed"}
word = "這肯定沒見過 "

# 若強行取值會報錯
try:
    vec = model.wv[word]
except KeyError as e:
    print(e)

# %% [markdown] papermill={"duration": 0.04768, "end_time": "2021-01-07T00:46:47.098084", "exception": false, "start_time": "2021-01-07T00:46:47.050404", "status": "completed"}
# ## 查看前 10 名相似詞

# %% [markdown] papermill={"duration": 0.046441, "end_time": "2021-01-07T00:46:47.191888", "exception": false, "start_time": "2021-01-07T00:46:47.145447", "status": "completed"}
# `model.wv.most_similar` 的 `topn` 預設為 10

# %% papermill={"duration": 7.430379, "end_time": "2021-01-07T00:46:54.669607", "exception": false, "start_time": "2021-01-07T00:46:47.239228", "status": "completed"}
model.wv.most_similar("飲料", topn=10)

# %% papermill={"duration": 0.158996, "end_time": "2021-01-07T00:46:54.877112", "exception": false, "start_time": "2021-01-07T00:46:54.718116", "status": "completed"}
model.wv.most_similar("car")

# %% papermill={"duration": 0.150343, "end_time": "2021-01-07T00:46:55.081792", "exception": false, "start_time": "2021-01-07T00:46:54.931449", "status": "completed"}
model.wv.most_similar("facebook")

# %% papermill={"duration": 0.156203, "end_time": "2021-01-07T00:46:55.289726", "exception": false, "start_time": "2021-01-07T00:46:55.133523", "status": "completed"}
model.wv.most_similar("詐欺")

# %% papermill={"duration": 0.146827, "end_time": "2021-01-07T00:46:55.487230", "exception": false, "start_time": "2021-01-07T00:46:55.340403", "status": "completed"}
model.wv.most_similar("合約")

# %% [markdown] papermill={"duration": 0.047968, "end_time": "2021-01-07T00:46:55.585374", "exception": false, "start_time": "2021-01-07T00:46:55.537406", "status": "completed"}
# ## 計算 Cosine 相似度

# %% papermill={"duration": 0.059902, "end_time": "2021-01-07T00:46:55.696853", "exception": false, "start_time": "2021-01-07T00:46:55.636951", "status": "completed"}
model.wv.similarity("連結", "鍵接")

# %% papermill={"duration": 0.060191, "end_time": "2021-01-07T00:46:55.808254", "exception": false, "start_time": "2021-01-07T00:46:55.748063", "status": "completed"}
model.wv.similarity("連結", "陰天")

# %% [markdown] papermill={"duration": 0.049844, "end_time": "2021-01-07T00:46:55.910099", "exception": false, "start_time": "2021-01-07T00:46:55.860255", "status": "completed"}
#  # 讀取模型

# %% papermill={"duration": 13.008536, "end_time": "2021-01-07T00:47:08.968071", "exception": false, "start_time": "2021-01-07T00:46:55.959535", "status": "completed"}
print(f"Loading {output_model}...")
new_model = word2vec.Word2Vec.load(output_model)

# %% papermill={"duration": 0.061289, "end_time": "2021-01-07T00:47:09.081548", "exception": false, "start_time": "2021-01-07T00:47:09.020259", "status": "completed"}
model.wv.similarity("連結", "陰天") == new_model.wv.similarity("連結", "陰天")
