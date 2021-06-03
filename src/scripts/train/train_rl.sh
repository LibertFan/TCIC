#!/usr/bin/env bash

export PRETRAINED_MODEL=/mnt/Projects/ImageTextMatching/log/TCIC/gen2m2_small_v1/checkpoint_last.pt
export SC_LR=1e-5
export VERSION=1
export LOG_MAIN_DIR=/mnt/Projects/ImageTextMatching
export FEATURES_DIR=/mnt/Projects/ImageTextMatching/data-bin
export CAPTIONS_DIR=/mnt/Projects/ImageTextMatching/data-bin
export SCENEGRAPH_DIR=/mnt/Projects/ImageTextMatching/data-bin
export LABEL_DIR=/mnt/Projects/ImageTextMatching/data-bin
export TREE_DIR=/mnt/Projects/ImageTextMatching/data-bin
export PROBLEM=TCIC_RL
export ARCH=gen2m2
export VALID_SUBSET=valid
export USER_DIR=./src
export TASK=scst_captioning
export MAX_SOURCE_POSITIONS=100
export MAX_TARGET_POSITIONS=100
export ENCODER_LAYERS=3
export DECODER_LAYERS=1
export SUPER_OBJ_NUM=16
export FEATURES=obj
export FEATURES_SPATIAL_ENCODING=--feature-spatial-encoding
export SC_CRITERION=base_scst
export SC_LABEL_SMOOTHING=0.0
export LR_SCHEDULER=polynomial_decay
export POWER=1.0
export END_LEARNING_RATE=0.0
export WARMUP_UPDATES=400
export MAX_UPDATE=40000
export OPTIMIZER=adam
export ADAM_EPS=1e-6
export DROPOUT=0.2
export ACTIVATION_DROPOUT=0.1
export ATTENTION_DROPOUT=0.1
export ADAM_BETAS=(0.9,0.98)
export WEIGHT_DECAY=1e-6
export CLIP_NORM=1.0
export SC_MAX_EPOCH=50
export MAX_TOKENS=4096
export UPDATE_FREQ=1
export LOG_INTERVAL=100
export KEEP_INTERVAL_UPDATE=200
export NUM_WORKERS=16
export SEED=1
export MAX_LEN=24
export MIN_LEN=1
export LEN_PEN=0.0
export NGRAM_SIZE=3
export SCR_BEAM=5
export SCB_BEAM=1
export CIDER_REWARD_WEIGHT=1.0
export BLEU_REWARD_WEIGHT=0.0
export CACHED_TOKENS=coco_train_idx
export SAVE_INTERVAL_UPDATES=500
export TEMPERATURE=1.0

SC_SAVE_DIR=${LOG_MAIN_DIR}/log/${PROBLEM}/${ARCH}_sc_v${VERSION}

echo PROBLEM: ${PROBLEM}
echo ARCH: ${ARCH}
echo SC_SAVE_DIR: ${SC_SAVE_DIR}

mkdir -p ${SC_SAVE_DIR}

if [[ ! -f "${SC_SAVE_DIR}/checkpoint_last.pt" ]]
then
    fairseq-train \
      --features-dir ${FEATURES_DIR} --captions-dir ${CAPTIONS_DIR} --sg-dir ${SCENEGRAPH_DIR} --save-dir ${SC_SAVE_DIR} \
      --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} --temperature ${TEMPERATURE} \
      --features ${FEATURES} --feature-spatial-encoding \
      --super-obj-num ${SUPER_OBJ_NUM} \
      --object-layernorm-embeddings \
      --max-source-positions ${MAX_SOURCE_POSITIONS} --max-target-positions ${MAX_TARGET_POSITIONS} \
      --encoder-layers ${ENCODER_LAYERS} --decoder-layers ${DECODER_LAYERS} \
      --lr ${SC_LR} --lr-scheduler polynomial_decay --power ${POWER} --end-learning-rate ${END_LEARNING_RATE} \
      --total-num-update ${MAX_UPDATE} --warmup-updates ${WARMUP_UPDATES} \
      --optimizer adam --adam-betas ${ADAM_BETAS} --weight-decay ${WEIGHT_DECAY} --adam-eps 1e-6 --clip-norm ${CLIP_NORM} \
      --criterion ${SC_CRITERION} --label-smoothing ${SC_LABEL_SMOOTHING} \
      --dropout ${DROPOUT} --attention-dropout ${ATTENTION_DROPOUT} --activation-dropout ${ACTIVATION_DROPOUT} \
      --max-epoch ${SC_MAX_EPOCH} --max-update ${MAX_UPDATE} --max-tokens ${MAX_TOKENS} --update-freq ${UPDATE_FREQ} \
      --log-format simple --log-interval ${LOG_INTERVAL} --num-workers ${NUM_WORKERS} \
      --save-interval-updates ${SAVE_INTERVAL_UPDATES} \
      --seed ${SEED} \
      --maximize-best-checkpoint-metric --best-checkpoint-metric cider \
      --cider-reward-weight ${CIDER_REWARD_WEIGHT} --bleu-reward-weight ${BLEU_REWARD_WEIGHT} --cached-tokens ${CACHED_TOKENS} \
      --scb-beam ${SCB_BEAM} --scr-beam ${SCR_BEAM} \
      --max-len-b ${MAX_LEN} --min-len ${MIN_LEN} --lenpen ${LEN_PEN} --no-repeat-ngram-size ${NGRAM_SIZE} --remove-bpe \
      --restore-file ${PRETRAINED_MODEL} \
      --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer \
      --tokenizer moses --bpe subword_nmt --bpe-codes ${CAPTIONS_DIR}/codes.txt \
      --valid-subset ${VALID_SUBSET} \
    | tee -a ${SC_SAVE_DIR}/train_log.txt
else
    fairseq-train \
      --features-dir ${FEATURES_DIR} --captions-dir ${CAPTIONS_DIR} --sg-dir ${SCENEGRAPH_DIR} --save-dir ${SC_SAVE_DIR} \
      --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} --temperature ${TEMPERATURE} \
      --features ${FEATURES} --feature-spatial-encoding \
      --super-obj-num ${SUPER_OBJ_NUM} \
      --object-layernorm-embeddings \
      --max-source-positions ${MAX_SOURCE_POSITIONS} --max-target-positions ${MAX_TARGET_POSITIONS} \
      --encoder-layers ${ENCODER_LAYERS} --decoder-layers ${DECODER_LAYERS} \
      --lr ${SC_LR} --lr-scheduler polynomial_decay --power ${POWER} --end-learning-rate ${END_LEARNING_RATE} \
      --total-num-update ${MAX_UPDATE} --warmup-updates ${WARMUP_UPDATES} \
      --optimizer adam --adam-betas ${ADAM_BETAS} --weight-decay ${WEIGHT_DECAY} --adam-eps 1e-6 --clip-norm ${CLIP_NORM} \
      --criterion ${SC_CRITERION} --label-smoothing ${SC_LABEL_SMOOTHING} \
      --dropout ${DROPOUT} --attention-dropout ${ATTENTION_DROPOUT} --activation-dropout ${ACTIVATION_DROPOUT} \
      --max-epoch ${SC_MAX_EPOCH} --max-update ${MAX_UPDATE} --max-tokens ${MAX_TOKENS} --update-freq ${UPDATE_FREQ} \
      --log-format simple --log-interval ${LOG_INTERVAL} --num-workers ${NUM_WORKERS} \
      --save-interval-updates ${SAVE_INTERVAL_UPDATES} \
      --seed ${SEED} \
      --maximize-best-checkpoint-metric --best-checkpoint-metric cider \
      --cider-reward-weight ${CIDER_REWARD_WEIGHT} --bleu-reward-weight ${BLEU_REWARD_WEIGHT} --cached-tokens ${CACHED_TOKENS} \
      --scb-beam ${SCB_BEAM} --scr-beam ${SCR_BEAM} \
      --max-len-b ${MAX_LEN} --min-len ${MIN_LEN} --lenpen ${LEN_PEN} --no-repeat-ngram-size ${NGRAM_SIZE} --remove-bpe \
      --tokenizer moses --bpe subword_nmt --bpe-codes ${CAPTIONS_DIR}/codes.txt \
      --valid-subset ${VALID_SUBSET} \
    | tee -a ${SC_SAVE_DIR}/train_log.txt
fi
