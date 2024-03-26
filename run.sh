# python run.py --gat_bert --embedding_type bert --output_dir data/output-gcn --dropout 0.3 --hidden_size 200 --learning_rate 5e-5 #R-GAT+BERT in restaurant
#python run.py --gat_bert --embedding_type bert --dataset_name laptop --output_dir data/output-gcn-laptop --dropout 0.3 --num_heads 7 --hidden_size 200 --learning_rate 5e-5 #R-GAT+BERT in laptop
#python run.py --gat_bert --embedding_type bert --dataset_name twitter --output_dir data/output-gcn-twitter --dropout 0.2  --hidden_size 200 --learning_rate 5e-5 #R-GAT+BERT in twitter
#python run.py --gat_our --highway --num_heads 7 --dropout 0.8 # R-GAT in restaurant
#python run.py --gat_our --dataset_name laptop --output_dir data/output-gcn-laptop --highway --num_heads 9 --per_gpu_train_batch_size 32 --dropout 0.7 --num_layers 3 --hidden_size 400 --final_hidden_size 400 # R-GAT in laptop
#python run.py --gat_our --dataset_name twitter --output_dir data/output-gcn-twitter --highway --num_heads 9 --per_gpu_train_batch_size 8 --dropout 0.6 --num_mlps 1 --final_hidden_size 400 # R-GAT in laptop


 # R-GAT+BERT in restaurant:  max f1: 0.8141 acc: 0.8696,  paper f1: 81.35 acc: 86.60  
# python run.py  --gat_bert   --cuda_id 0 --embedding_type bert --output_dir data/output-gcn-rest --dropout 0.3 --hidden_size 200 --learning_rate 5e-5  --glove_dir /nas2/lishengping/models/word2vec_glove/glove.840B.300d.txt  --dataset_name rest --bert_model_dir /nas2/archived/qsj/bert-model/bert-base-uncased 

 
 # R-GAT+BERT in laptop:  max f1: 0.7541 acc: 0.7931  paper  f1: 74.07  acc: 78.21 
# python run.py  --gat_bert   --cuda_id 1 --embedding_type bert --output_dir data/output-gcn-laptop --dropout 0.3 --hidden_size 200 --learning_rate 5e-5  --glove_dir /nas2/lishengping/models/word2vec_glove/glove.840B.300d.txt  --dataset_name laptop --bert_model_dir /nas2/archived/qsj/bert-model/bert-base-uncased  --num_heads 7

 # R-GAT+BERT in twitter:   max f1: 0.7352 acc: 0.7486  paper  f1: 74.88  acc: 76.15 
CUDA_VISIBLE_DEVICES=2 python run.py  --gat_bert   --cuda_id 2 --embedding_type bert --output_dir data/output-gcn-twitter2 --dropout 0.2 --hidden_size 200 --learning_rate 5e-5  --glove_dir /nas2/lishengping/models/word2vec_glove/glove.840B.300d.txt  --dataset_name twitter --bert_model_dir /nas2/archived/qsj/bert-model/bert-base-uncased --num_heads 6