clip_benchmark eval \
    --model_type "mobileMLiT" \
    --model "mobileclip_s1" \
    --dataset "webdatasets.txt" \
    --batch_size 128 \
    --image_encoder_id "nvidia/MambaVision-L-1K" \
    --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" \
    --output "/root/code/harim/DeepDaiv/CLIP_benchmark/output/benchmark_mobileclip_{dataset}_{pretrained}_{model}_{language}_{task}.json"