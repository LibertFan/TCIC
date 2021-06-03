#!/usr/bin/env bash


export PYTHONPATH=./supports/pycocoeval:$PYTHONPATH


function gen_func {
    MODEL_DIR=$1
    BATCH_SIZE=$2
    BEAM_SIZE=$3
    MIN_LEN=$4
    MAX_LEN=$5
    NGRAM_SIZE=$6
    LEN_PEN=$7
    GPU_ID=$8
    CKPT=$9
    SUBSET=${10}

    CKPT_ID=$(echo ${CKPT} | sed 's/checkpoint//g' | sed 's/\.pt//g' | sed 's/^_//g')

    MODEL=${MODEL_DIR}/${CKPT}
    RESULT_DIR=${MODEL_DIR}/${SUBSET}/BEAMSIZE${BEAM_SIZE}_MINLEN${MIN_LEN}_MAXLEN${MAX_LEN}_NGRAMSIZE${NGRAM_SIZE}_LENPEN${LEN_PEN}/${CKPT_ID}

    mkdir -p ${RESULT_DIR}

    echo GPU_ID: ${GPU_ID}. CKPT: ${CKPT}. CKPT_ID: ${CKPT_ID}. \
        USER_DIR: ${USER_DIR}. BATCH_SIZE: ${BATCH_SIZE}. BEAM: ${BEAM_SIZE}. MIN_LEN: ${MIN_LEN}. \
        NGRAM_SIZE: ${NGRAM_SIZE}. LEN_PEN: ${LEN_PEN}

    if [[ ! -f "${RESULT_DIR}/hypo.txt" ]] || [[ $((5000)) -gt `wc -l ${RESULT_DIR}/hypo.txt | awk '{ print $1 }'` ]]
    then
        CUDA_VISIBLE_DEVICES=${GPU_ID} python ${USER_DIR}/generate.py --skip-invalid-size-inputs-valid-test \
            --path ${MODEL} \
            --batch-size ${BATCH_SIZE} --beam ${BEAM_SIZE} --max-len-b ${MAX_LEN} --min-len ${MIN_LEN} \
            --lenpen ${LEN_PEN} --no-repeat-ngram-size ${NGRAM_SIZE} \
            --user-dir ${USER_DIR} --task ${TASK} --features ${FEATURES} \
            --features-dir ${FEATURES_DIR} --captions-dir ${CAPTIONS_DIR} --sg-dir ${SCENEGRAPH_DIR} --source-only \
            --gen-subset ${SUBSET} --tokenizer moses --bpe subword_nmt --bpe-codes ${CAPTIONS_DIR}/codes.txt \
            > ${RESULT_DIR}/res.txt

        cat ${RESULT_DIR}/res.txt | grep 'H-' | cut -f3- | sed 's/ ##//g' > ${RESULT_DIR}/hypo.txt

    fi

    python ${USER_DIR}/scripts/support/raw2json.py --input ${RESULT_DIR}/res.txt --output ${RESULT_DIR}/hypo.json

    python ${USER_DIR}/scripts/support/score.py \
        --reference-captions ${CAPTIONS_DIR}/annotations/captions_val2014.json \
        --system-captions ${RESULT_DIR}/hypo.json ${NO_SPICE} > ${RESULT_DIR}/result.txt

    cat ${RESULT_DIR}/result.txt
    # cat ${RESULT_DIR}/result.txt > ${MODEL_DIR}/${SUBSET}/best_result.txt
    python ${USER_DIR}/scripts/support/write_record.py --result_path ${RESULT_DIR}/result.txt \
        --ckpt ${CKPT} --beam ${BEAM_SIZE} --min_len ${MIN_LEN} --max_len ${MAX_LEN} --ngram_size ${NGRAM_SIZE} \
        --len_pen ${LEN_PEN} --save_path ${MODEL_DIR}/${SUBSET}/best.json
}

MODEL_DIR=$1
BATCH_SIZE=$2
BEAM_SIZE=$3
MIN_LEN=$4
MAX_LEN=$5
NGRAM_SIZE=$6
LEN_PEN=$7
GPU_ID=$8
CKPT=$9

gen_func ${MODEL_DIR} ${BATCH_SIZE} ${BEAM_SIZE} ${MIN_LEN} ${MAX_LEN} ${NGRAM_SIZE} ${LEN_PEN} ${GPU_ID} ${CKPT} test

