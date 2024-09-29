import torch
import os
import mobileclip
from transformers import AutoModel

def change_image_encoder(model, image_encoder_id, device):
    
    print("Changing image encoder to", image_encoder_id)
    model.eval()
    model.visual = None
    torch.cuda.empty_cache()

    print("Lock image encoder weights and train only text encoder")

    model.visual = AutoModel.from_pretrained(image_encoder_id, trust_remote_code=True).model

    encoder_output_dims = {
        "nvidia/MambaVision-T-1K": 640,
        "nvidia/MambaVision-B-1K": 1024,
        "nvidia/MambaVision-S-1K": 768,
        "nvidia/MambaVision-L-1K": 1568,
        "nvidia/MambaVision-L2-1K": 1640
    }
    if image_encoder_id in encoder_output_dims:
        model.visual.head = torch.nn.Linear(encoder_output_dims[image_encoder_id], 512)
    else:
        raise ValueError(f"지원되지 않는 이미지 인코더입니다: {image_encoder_id}")
    
    
    model = model.to(device)

    for param in model.visual.parameters():
        param.requires_grad = False
    for param in model.visual.head.parameters():
        param.requires_grad = True
    return model


def load_mobile_MLiT(model_name: str = "mobileclip_s1", pretrained: str = None, cache_dir: str = None, device="cpu", image_encoder_id:str = 'nvidia/MambaVision-T-1K'):
   
    print(f"모델 이름: {model_name}")
    print(f"사용 장치: {device}")
    print(f"이미지 인코더: {image_encoder_id}")
    
    try:
        if pretrained:
            model, _, transform = mobileclip.create_model_and_transforms(model_name, pretrained=pretrained)
            
            if image_encoder_id:
                model = change_image_encoder(model, image_encoder_id, device)
            
            
        else:
            print("사전 훈련된 가중치 없이 모델을 생성합니다.")
            
            model, _, transform = mobileclip.create_model_and_transforms(model_name)
            
            
            if image_encoder_id:
                model = change_image_encoder(model, image_encoder_id, device)
            

        model = model.to(device)
        print(f"구조 {dir(model)}")
        
        
        tokenizer = mobileclip.get_tokenizer(model_name)

        print(f"모델 로딩 성공: {model.__class__.__name__}")
        return model, transform, tokenizer
    except Exception as e:
        print(f"MobileMLiT 모델 로딩 중 오류 발생: {str(e)}")
        raise
    return None