python -u train_spikformer.py \
        --seed 42 \
        --dataset_name mr \
        --label_num 2 \
        --batch_size 12 \
        --fine_tune_lr 1e-4 \
        --epochs 100 \
        --depths 6 \
        --max_length 64 \
        --dim 768 \
        --tau 10.0 \
        --common_thr 1.0 \
        --num_step 32 \
        --tokenizer_path "saved_models/bert-base-cased" \
        > "train.log"