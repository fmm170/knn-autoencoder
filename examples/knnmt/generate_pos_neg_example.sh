CUDA_VISIBLE_DEVICES=0
domain="train_ae_medical"
data_bin=./multi-domain-data-bin/${domain}
datastore=./multi-domain-datastore/${domain}
datastore_size=164000

num_neighbors=2
pos_sample=/users10/lhuang/xiaokenaifan/ConAE/train/MSMARCO
neg_sample=/users10/lhuang/xiaokenaifan/ConAE/train/MSMARCO

checkpoint=/users10/lhuang/xiaokenaifan/knn-models/examples/knnmt/pretrained_model/wmt19.de-en.ffn8192.pt

# mkdir -p ${datastore}

python /users10/lhuang/xiaokenaifan/knn-models/knn_models_cli/generate_mt_datastore.py ${data_bin} \
    --task translation_knn \
    --gen-subset train \
    --path ${checkpoint} \
    --datastore ${datastore} \
    --datastore-size ${datastore_size} \
    --keys-dtype fp16 \
    --max-tokens 8000 \
    --num-neighbors ${num_neighbors} \
    --pos-sample ${pos_sample} \
    --neg-sample ${neg_sample} \
    --keys-dimension 1024 \
    --whether-generate-datastore \
    --saving-mode