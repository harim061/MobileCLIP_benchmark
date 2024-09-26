import torch
import mobileclip
import os

def load_mobile_clip(model_name: str = "mobileclip_s0", pretrained: str = None, cache_dir: str = None, device="cpu"):

    pretrained = f'./clip_benchmark/models/checkpoints/{model_name}.pt'
    print(f"모델 이름: {model_name}")
    print(f"Pretrained 가중치: {pretrained}")
    print(f"Cache 디렉토리: {cache_dir}")
    print(f"사용 장치: {device}")
    try:
        if pretrained:
            print(f"체크포인트를 로드하려고 시도 중: {pretrained}")
            print(f"파일 존재 여부: {os.path.exists(pretrained)}")
            model, _, transform = mobileclip.create_model_and_transforms(model_name, pretrained=pretrained)
        else:
            print("사전 훈련된 가중치 없이 모델을 생성합니다.")
            model, _, transform = mobileclip.create_model_and_transforms(model_name)

        model = model.to(device)
        print(f"구조 {dir(model)}")
        
        tokenizer = mobileclip.get_tokenizer(model_name)

        print(f"모델 로딩 성공: {model.__class__.__name__}")
        return model, transform, tokenizer
    except Exception as e:
        print(f"MobileCLIP 모델 로딩 중 오류 발생: {str(e)}")
        raise