#!/usr/bin/env bash

cd ..

 python main.py \
    -b 64 \
   --cuda-device 0 \
    train \
   --tacred ../../datasets/TACRED_data/data_tacred \
   -pp /fs/scratch/rng_cr_bcai_dl_students/r26/huggingface-transformers/roberta-base \
   --print-every 30 \
   -e 7 \
   -rl "Testing EMT and ESS" \
   -el "Enriched Attention on PLM (GPU Side)" \
   Enriched_Attention
