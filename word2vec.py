from gensim.models import word2vec
import multiprocessing

WIKI_SEG_TXT = "wiki_seg.txt"
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
