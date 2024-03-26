
## 准备
### Glove 模型下载
`glove.840B.300d.zip`
下载链接：https://nlp.stanford.edu/projects/glove/

### BERT预训练模型
`bert-base-uncased`

### 环境安装
`pip install -r requirements.txt`

## 数据处理
### 依存分析
依存分析模型：https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz

下载后放到 models/ 中

`python data_preprocess_semeval.py`

`python data_preprocess_twitter.py`


已经处理好的依存分析结果在：`./data/`


## 训练
`bash run.sh`


## 结果
| model   | Twitter    | Rest | Laptop
| ------ | ---- | ------ | ---- |
|    | acc /  f1    | acc /  f1| acc /  f1|
| ASGCN   | 71.53 /  69.68    | 74.14 / 69.24 | 80.86 / 72.19 |
| AWIGCN   | 72.98 / 71.40   |74.92 / 70.46    | 81.62 / 73.51|
| BERT   | 75.28 / 74.11   |85.62 / 78.28    | 77.58 / 72.38|
| GAT_BERT   |  76.15 / 74.88 | 86.60 / 81.35  |78.21 / 74.07|
| OURS   | 74.86 / 73.52 | 86.96 / 81.41  |79.31 / 75.41|
