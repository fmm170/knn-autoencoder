export PCKMT_DATASTORE="1"
CUDA_VISIBLE_DEVICES=0

domain="medical"
data_bin=/users10/lhuang/xiaokenaifan/knn-models/examples/knnmt/multi-domain-data-bin/${domain}
datastore=./multi-domain-datastore/${domain}
datastore_size=6501418
checkpoint=/users10/lhuang/xiaokenaifan/knn-models/examples/knnmt/pretrained_model/wmt19.de-en.ffn8192.pt

mkdir -p ${datastore}

python /users10/lhuang/xiaokenaifan/knn-models/knn_models_cli/generate_mt_datastore_my.py ${data_bin} \
    --task translation_knn \
    --gen-subset train \
    --path ${checkpoint} \
    --datastore ${datastore} \
    --datastore-size ${datastore_size} \
    --keys-dtype fp16 \
    --max-tokens 8000 \
    --whether-generate-datastore \
    --saving-mode