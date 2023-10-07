python predistill_spikformer.py \
       --seed 42 \
       --batch_size 16 \
       --max_sample_num 80462898 \
       --fine_tune_lr 5e-5 \
       --epochs 1 \
       --teacher_model_path "saved_models/bert-base-cased" \
       --depths 6 \
       --max_length 256 \
       --dim 768 \
       --rep_weight 1 \
       --num_step 16 \
       > "distill_logs/predistill.log"