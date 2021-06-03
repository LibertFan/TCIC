#!/usr/bin/env bash

ROOT_DIR=.

export LR=5e-4
export VERSION=1
export LOG_MAIN_DIR=${ROOT_DIR}
export FEATURES_DIR=${ROOT_DIR}/data-bin
export CAPTIONS_DIR=${ROOT_DIR}/data-bin
export SCENEGRAPH_DIR=${ROOT_DIR}/data-bin
export PROBLEM=TCIC
export ARCH=gen2m2_small
export USER_DIR=./src
export TASK=captioning
export SUPER_OBJ_NUM=16
export MAX_SOURCE_POSITIONS=100
export MAX_TARGET_POSITIONS=128
export ENCODER_LAYERS=3
export DECODER_LAYERS=1
export FEATURES=obj
export FEATURES_SPATIAL_ENCODING=--feature-spatial-encoding
export CRITERION=loss_m17
export LABEL_SMOOTHING=0.2
export LAMBDA_SUPER=0.1
export LAMBDA_ALIGN=100.0
export LR_SCHEDULER=polynomial_decay
export POWER=1.0
export END_LEARNING_RATE=0.0
export WARMUP_UPDATES=1000
export MAX_UPDATE=10000
export OPTIMIZER=adam
export ADAM_EPS=1e-6
export DROPOUT=0.3
export ACTIVATION_DROPOUT=0.1
export ATTENTION_DROPOUT=0.1
export ADAM_BETAS=(0.9,0.98)
export WEIGHT_DECAY=1e-6
export CLIP_NORM=1.0
export MAX_EPOCH=50
export MAX_TOKENS=4096
export UPDATE_FREQ=8
export LOG_INTERVAL=20
export KEEP_INTERVAL_UPDATE=200
export NUM_WORKERS=16
export SEED=1


SAVE_DIR=${LOG_MAIN_DIR}/log/${PROBLEM}/${ARCH}_v${VERSION}

echo PROBLEM: ${PROBLEM}
echo ARCH: ${ARCH}
echo SAVE_DIR: ${SAVE_DIR}

mkdir -p ${SAVE_DIR}

fairseq-train \
  --features-dir ${FEATURES_DIR} \
  --captions-dir ${CAPTIONS_DIR} \
  --sg-dir ${SCENEGRAPH_DIR} \
  --save-dir ${SAVE_DIR} \
  --user-dir ${USER_DIR} \
  --task ${TASK} \
  --arch ${ARCH} \
  --features ${FEATURES} \
  --feature-spatial-encoding \
  --super-obj-num ${SUPER_OBJ_NUM} \
  --object-layernorm-embeddings \
  --max-source-positions ${MAX_SOURCE_POSITIONS} \
  --max-target-positions ${MAX_TARGET_POSITIONS} \
  --encoder-layers ${ENCODER_LAYERS} \
  --decoder-layers ${DECODER_LAYERS} \
  --lr ${LR} \
  --lr-scheduler polynomial_decay \
  --power ${POWER} \
  --end-learning-rate ${END_LEARNING_RATE} \
  --total-num-update ${MAX_UPDATE} \
  --warmup-updates ${WARMUP_UPDATES} \
  --optimizer adam \
  --adam-betas ${ADAM_BETAS} \
  --weight-decay ${WEIGHT_DECAY} \
  --adam-eps 1e-6 \
  --clip-norm 0.1 \
  --criterion loss_tcic \
  --label-smoothing ${LABEL_SMOOTHING} \
  --lambda-super ${LAMBDA_SUPER} \
  --lambda-align ${LAMBDA_ALIGN} \
  --dropout ${DROPOUT} \
  --attention-dropout ${ATTENTION_DROPOUT} \
  --activation-dropout ${ACTIVATION_DROPOUT} \
  --max-epoch ${MAX_EPOCH} \
  --max-update ${MAX_UPDATE} \
  --max-tokens ${MAX_TOKENS} \
  --update-freq ${UPDATE_FREQ} \
  --log-format simple \
  --log-interval ${LOG_INTERVAL} \
  --num-workers ${NUM_WORKERS} \
  --seed ${SEED} \
| tee -a ${SAVE_DIR}/train_log.txt
