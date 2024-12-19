for i in $(seq -w 0 29); do
    python tools/preprocess_data.py \
        --input ../pile/${i}.jsonl \
        --output-prefix ../pile_gpt_test/${i} \
        --vocab-file ../gpt2-vocab.json \
        --tokenizer-type GPT2BPETokenizer \
        --merge-file ../gpt2-merges.txt \
        --append-eod \
        --workers 32
done