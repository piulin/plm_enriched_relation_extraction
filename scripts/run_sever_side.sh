#!/usr/bin/env bash

cd ..

 python main.py \
   --tacred ../../datasets/TACRED_data/data_tacred \
   --cuda-device 0 \
   -pp /fs/scratch/rng_cr_bcai_dl_students/r26/huggingface-transformers/roberta-base \
   --print-every 30 \
   -b 64 \
   -e 7 \
   -rl "Testing EMT and ESS" \
   -el "Enriched Attention on PLM (GPU Side)"