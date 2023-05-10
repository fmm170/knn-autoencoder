CUDA_VISIBLE_DEVICES=0
domain="medical"
data_bin=./multi-domain-data-bin/${domain}
datastore=./multi-domain-datastore/iwslt_medical_1024
datastore_size=6501418
checkpoint=/users10/lhuang/xiaokenaifan/fairseq/examples/translation/checkpoints/checkpoint3.pt
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