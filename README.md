
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
`data_preprocess_twitter.py`

已经处理好的依存分析结果在：`./data/`


## 训练
`bash run.sh`


