clip_benchmark eval \
    --model_type "mobile_clip" \
    --model "mobileclip_s1" \
    --dataset "webdatasets.txt" \
    --batch_size 128 \
    --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" \
    --output "/root/code/harim/DeepDaiv/CLIP_benchmark/output_m1/benchmark_mobileclip_{dataset}_{pretrained}_{model}_{language}_{task}.json"