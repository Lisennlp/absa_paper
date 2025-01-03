
# rest best train args
#  # R-GAT+BERT in restaurant:  max f1:  80.12 acc: 86.43,  paper f1: 81.35 acc: 86.60  
# CUDA_VISIBLE_DEVICES=0 python run.py  --gat_bert   --cuda_id 0 --embedding_type bert --output_dir data/output-gcn-rest-best --dropout 0.3 --hidden_size 200 --learning_rate 5e-5  --glove_dir /nas2/lishengping/models/word2vec_glove/glove.840B.300d.txt  --dataset_name rest --bert_model_dir /nas2/archived/qsj/bert-model/bert-base-uncased --balanced_data
 # laptop
#  # R-GAT+BERT in laptop:  max f1: 0.7541 acc: 0.7931  paper  f1: 74.07  acc: 78.21 
# CUDA_VISIBLE_DEVICES=1 python run.py  --gat_bert   --cuda_id 1 --embedding_type bert --output_dir data/output-gcn-laptop-best --dropout 0.3 --hidden_size 200 --learning_rate 5e-5  --glove_dir /nas2/lishengping/models/word2vec_glove/glove.840B.300d.txt  --dataset_name laptop --bert_model_dir /nas2/archived/qsj/bert-model/bert-base-uncased  --num_heads 7
# twitter
#  # R-GAT+BERT in twitter:   max f1: 74.39 acc: 75.43  paper  f1: 74.88  acc: 76.15 
# CUDA_VISIBLE_DEVICES=2 python run.py  --gat_bert   --cuda_id 3 --embedding_type bert --output_dir data/output-gcn-twitter-best --dropout 0.3 --hidden_size 200 --learning_rate 5e-5  --glove_dir /nas2/lishengping/models/word2vec_glove/glove.840B.300d.txt  --dataset_name twitter --bert_model_dir /nas2/archived/qsj/bert-model/bert-base-uncased --num_heads 7 --balanced_data


# # test train args  rest
# CUDA_VISIBLE_DEVICES=0 python run.py  --gat_bert   --cuda_id 0 --embedding_type bert --output_dir data/output-gcn-rest-test1 --dropout 0.3 --hidden_size 200 --learning_rate 1e-5  --glove_dir /nas2/lishengping/models/word2vec_glove/glove.840B.300d.txt  --dataset_name rest --bert_model_dir /nas2/archived/qsj/bert-model/bert-base-uncased --balanced_data

 # laptop
# CUDA_VISIBLE_DEVICES=1 python run.py  --gat_bert   --cuda_id 1 --embedding_type bert --output_dir data/output-gcn-laptop-test --dropout 0.3 --hidden_size 200 --learning_rate 5e-5  --glove_dir /nas2/lishengping/models/word2vec_glove/glove.840B.300d.txt  --dataset_name laptop --bert_model_dir /nas2/archived/qsj/bert-model/bert-base-uncased  --num_heads 7

# # twitter
# CUDA_VISIBLE_DEVICES=2 python run.py  --gat_bert   --cuda_id 3 --embedding_type bert --output_dir data/output-gcn-twitter-test --dropout 0.3 --hidden_size 200 --learning_rate 5e-5  --glove_dir /nas2/lishengping/models/word2vec_glove/glove.840B.300d.txt  --dataset_name twitter --bert_model_dir /nas2/archived/qsj/bert-model/bert-base-uncased --num_heads 7 --balanced_data


# absa
# CUDA_VISIBLE_DEVICES=3 python run.py  --gat_bert   --cuda_id 0 --embedding_type xlnet --output_dir data/output-gcn-rest-test1 --dropout 0.2 --hidden_size 512 --learning_rate 2e-5  --glove_dir /nas2/lishengping/models/word2vec_glove/glove.840B.300d.txt  --dataset_name rest --bert_model_dir /nas2/lishengping/models/pretrain_models/xlnet-large-cased  --xlnet_cnn_lstm --per_gpu_train_batch_size 8 --gradient_accumulation_steps 4 2>&1 |tee xlnet_cnn_lstm_argu.b8x4.lr2e-5.d0.2log

CUDA_ID=0
LR=1e-05
DROPOUT=0.2
BATCH_SIZE=8
GAS=1
DATASET_NAME=coffe
TENSORBOARD_SUFFIX=''
CUDA_VISIBLE_DEVICES=$CUDA_ID python run.py  --gat_bert  --cuda_id 0 --embedding_type xlnet --output_dir data/output-gcn-rest-test1 --dropout $DROPOUT --hidden_size 512 --learning_rate $LR  --glove_dir /nas2/lishengping/models/word2vec_glove/glove.840B.300d.txt  --dataset_name $DATASET_NAME --bert_model_dir /nas2/lishengping/models/pretrain_models/xlnet-large-cased  --xlnet_cnn_lstm --per_gpu_train_batch_size $BATCH_SIZE --gradient_accumulation_steps $GAS --tensorboard_dir xlnet_cnn_lstm_tensorboard/$DATASET_NAME/B$BATCH_SIZE'x'$GAS.LR$LR.D$DROPOUT$TENSORBOARD_SUFFIX 2>&1 | tee logs/xlnet_cnn_lstm.$DATASET_NAME.B$BATCH_SIZE'x'$GAS.LR$LR.D$DROPOUT.log

# error: allennlp.common.checks.ConfigurationError: universal_dependencies not in acceptable choices for dataset_reader.type: ['babi', 'conll2003', 'interleaving', 'multitask', 'multitask_shim', 'sequence_tagging', 'sharded', 'text_classification_json']. You should either use the --include-package flag to make sure the correct module is loaded, or use a fully qualified class name in your config file like {"model": "my_module.models.MyModel"} to have it imported automatically.
# fix: /nas2/lishengping/miniconda3/envs/T1.12C11.7/bin/pip install allennlp-models