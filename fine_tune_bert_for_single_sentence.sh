python -u fine_tune_bert_for_single_sentence.py \
    --dataset_name "sst2" \
    --batch_size 128 \
    --fine_tune_lr 5e-5 \
    --epochs 10 \
    --teacher_model_name "bert-base-cased" \
    --label_num 2 \
    --metric "acc" > "fine_tune_bert.log"