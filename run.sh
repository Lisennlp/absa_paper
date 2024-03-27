
#  # R-GAT+BERT in restaurant:  max f1:  80.12 acc: 86.43,  paper f1: 81.35 acc: 86.60  
# CUDA_VISIBLE_DEVICES=0 python run.py  --gat_bert   --cuda_id 0 --embedding_type bert --output_dir data/output-gcn-rest-final4_b --dropout 0.3 --hidden_size 200 --learning_rate 5e-5  --glove_dir /nas2/lishengping/models/word2vec_glove/glove.840B.300d.txt  --dataset_name rest --bert_model_dir /nas2/archived/qsj/bert-model/bert-base-uncased --balanced_data

 
#  # R-GAT+BERT in laptop:  max f1: 0.7541 acc: 0.7931  paper  f1: 74.07  acc: 78.21 
# CUDA_VISIBLE_DEVICES=1 python run.py  --gat_bert   --cuda_id 1 --embedding_type bert --output_dir data/output-gcn-laptop-final --dropout 0.3 --hidden_size 200 --learning_rate 5e-5  --glove_dir /nas2/lishengping/models/word2vec_glove/glove.840B.300d.txt  --dataset_name laptop --bert_model_dir /nas2/archived/qsj/bert-model/bert-base-uncased  --num_heads 7

#  # R-GAT+BERT in twitter:   max f1: 74.39 acc: 75.43  paper  f1: 74.88  acc: 76.15 
# CUDA_VISIBLE_DEVICES=2 python run.py  --gat_bert   --cuda_id 3 --embedding_type bert --output_dir data/output-gcn-twitter4 --dropout 0.3 --hidden_size 200 --learning_rate 5e-5  --glove_dir /nas2/lishengping/models/word2vec_glove/glove.840B.300d.txt  --dataset_name twitter --bert_model_dir /nas2/archived/qsj/bert-model/bert-base-uncased --num_heads 7 --balanced_data



# CUDA_VISIBLE_DEVICES=0 python run.py  --gat_bert   --cuda_id 0 --embedding_type bert --output_dir data/output-gcn-rest-final4_b --dropout 0.3 --hidden_size 200 --learning_rate 5e-5  --glove_dir /nas2/lishengping/models/word2vec_glove/glove.840B.300d.txt  --dataset_name rest --bert_model_dir /nas2/archived/qsj/bert-model/bert-base-uncased --balanced_data

 
# CUDA_VISIBLE_DEVICES=1 python run.py  --gat_bert   --cuda_id 1 --embedding_type bert --output_dir data/output-gcn-laptop-final --dropout 0.3 --hidden_size 200 --learning_rate 5e-5  --glove_dir /nas2/lishengping/models/word2vec_glove/glove.840B.300d.txt  --dataset_name laptop --bert_model_dir /nas2/archived/qsj/bert-model/bert-base-uncased  --num_heads 7

# CUDA_VISIBLE_DEVICES=2 python run.py  --gat_bert   --cuda_id 3 --embedding_type bert --output_dir data/output-gcn-twitter4 --dropout 0.3 --hidden_size 200 --learning_rate 5e-5  --glove_dir /nas2/lishengping/models/word2vec_glove/glove.840B.300d.txt  --dataset_name twitter --bert_model_dir /nas2/archived/qsj/bert-model/bert-base-uncased --num_heads 7 --balanced_data