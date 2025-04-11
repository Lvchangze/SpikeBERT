$fine_tuned_bert_path=XXX
$predistill_model_path=XXX

python -u new_distill_spikformer.py \
        --seed 42 \
        --dataset_name sst2 \
        --data_augment "False" \
        --label_num 2 \
        --batch_size 32 \
        --fine_tune_lr 5e-4 \
        --epochs 30 \
        --teacher_model_path ${fine_tuned_bert_path} \
        --depths 12 \
        --max_length 128 \
        --dim 768 \
        --ce_weight 0.1 \
        --emb_weight 0.1 \
        --logit_weight 1.0 \
        --rep_weight 0.1 \
        --tau 2.0 \
        --common_thr 1.0 \
        --predistill_model_path ${predistill_model_path} \
        --num_step 4 \
        --ignored_layers 0 \
        --metric "acc" \
        > "distill_logs/sst2.log"
