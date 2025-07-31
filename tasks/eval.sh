#!/bin/bash
# evaluate_all_tasks.sh

# 设置环境变量
export CUDA_VISIBLE_DEVICES=3
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=localhost
export MASTER_PORT=29582

# 模型和数据路径
# /home/ubuntu/tiansheng/26_ICLR_btk_moe/ReMoE-ts/logs/MOE-Topk-182M-0723-183105
# /home/ubuntu/tiansheng/26_ICLR_btk_moe/ReMoE-ts/logs/MOE-SPTopk-182M-0725-003932
# /home/ubuntu/tiansheng/26_ICLR_btk_moe/ReMoE-ts/logs/MOE-SPTopk-G8-182M-0726-042101
MODEL_PATH="/home/ubuntu/tiansheng/26_ICLR_btk_moe/ReMoE-ts/logs/MOE-SPTopk-182M-0725-003932"
LAMBADA_DATA="/home/ubuntu/tiansheng/26_ICLR_btk_moe/ReMoE-ts/tasks/data/lambada_test.jsonl"
MNLI_DATA="/path/to/mnli_dev.tsv"
QQP_DATA="/path/to/qqp_dev.tsv"
RACE_DATA="/home/ubuntu/tiansheng/26_ICLR_btk_moe/ReMoE-ts/tasks/data/RACE"
ARC_E_DATA="/home/ubuntu/tiansheng/26_ICLR_btk_moe/ReMoE-ts/tasks/data/arc/ARC-Easy"
ARC_C_DATA="/home/ubuntu/tiansheng/26_ICLR_btk_moe/ReMoE-ts/tasks/data/arc/ARC-Challenge"

# MOE配置参数（与训练时保持一致）
NUM_EXPERTS=8
NUM_TOPK=6
GRANULARITY=1

# 通用模型参数（基于训练脚本）
MODEL_ARGS="\
--use-mcore-models \
--disable-bias-linear \
--num-layers 12 \
--hidden-size 768 \
--ffn-hidden-size 3072 \
--num-attention-heads 12 \
--max-position-embeddings 1024 \
--init-method-std 0.01 \
--attention-dropout 0.0 \
--hidden-dropout 0.0 \
--normalization RMSNorm \
--position-embedding-type rope \
--swiglu \
--untie-embeddings-and-output-weights \
--group-query-attention \
--num-query-groups 4 \
--no-masked-softmax-fusion \
--no-position-embedding \
--rotary-base 1000000 \
--use-flash-attn \
--tokenizer-type GPT2BPETokenizer \
--vocab-file /home/ubuntu/tiansheng/26_ICLR_btk_moe/ReMoE-ts/gpt2-vocab.json \
--merge-file /home/ubuntu/tiansheng/26_ICLR_btk_moe/ReMoE-ts/gpt2-merges.txt \
--make-vocab-size-divisible-by 1024 \
--bf16 \
--no-load-optim \
--no-load-rng"

# MOE特定参数    --moe-sample-routing \
MOE_ARGS="\
--num-experts $NUM_EXPERTS \
--moe-router-topk $NUM_TOPK \
--moe-router-load-balancing-type aux_loss \
--moe-aux-loss-coeff 1e-2 \
--moe-token-dispatcher-type alltoall \
--overlap-param-gather \
--overlap-grad-reduce \
--moe-router-pre-softmax \
--moe-grouped-gemm \
--moe-layer-recompute \
--moe-sample-routing \
--attention-softmax-in-fp32 \
--moe-granularity $GRANULARITY \
--moe-router-eval-topk 2 4 6"

# 模型并行参数
MODEL_PARALLEL_ARGS="\
--tensor-model-parallel-size 1 \
--pipeline-model-parallel-size 1 \
--expert-model-parallel-size 1 \
--use-distributed-optimizer \
--sequence-parallel"

echo "=== Starting Zero-shot Evaluation with MOE Model ==="
echo "MOE Configuration: $NUM_EXPERTS experts, top-$NUM_TOPK routing, granularity $GRANULARITY"

# 1. LAMBADA评估
# echo "1. Evaluating LAMBADA..."
# echo "Model path: $MODEL_PATH"
# echo "Data path: $LAMBADA_DATA"
# python main.py \
#     --task LAMBADA \
#     --valid-data $LAMBADA_DATA \
#     --load $MODEL_PATH \
#     --seq-length 1024 \
#     --micro-batch-size 8 \
#     --log-interval 100 \
#     $MODEL_ARGS \
#     $MOE_ARGS \
#     $MODEL_PARALLEL_ARGS 2>&1 | tee lambada_eval.log

# 2. RACE评估
# echo "2. Evaluating RACE..."
# python main.py \
#     --task RACE \
#     --valid-data $RACE_DATA \
#     --load $MODEL_PATH \
#     --epochs 0 \
#     --seq-length 1024 \
#     --micro-batch-size 8 \
#     $MODEL_ARGS \
#     $MOE_ARGS \
#     $MODEL_PARALLEL_ARGS

# echo "3. Evaluating ARC-C..."
python main.py \
    --task ARC-C \
    --valid-data $ARC_C_DATA \
    --load $MODEL_PATH \
    --epochs 0 \
    --seq-length 1024 \
    --micro-batch-size 8 \
    $MODEL_ARGS \
    $MOE_ARGS \
    $MODEL_PARALLEL_ARGS

# echo "4. Evaluating ARC-E..."
# python main.py \
#     --task ARC-E \
#     --valid-data $ARC_E_DATA \
#     --load $MODEL_PATH \
#     --epochs 0 \
#     --seq-length 1024 \
#     --micro-batch-size 8 \
#     $MODEL_ARGS \
#     $MOE_ARGS \
#     $MODEL_PARALLEL_ARGS

echo "=== Zero-shot Evaluation Complete ==="