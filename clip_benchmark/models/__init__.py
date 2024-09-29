from typing import Union
import torch
from .open_clip import load_open_clip
from .japanese_clip import load_japanese_clip
from .mobile_clip import load_mobile_clip
from .mobileMLiT import load_mobile_MLiT

# loading function must return (model, transform, tokenizer)
TYPE2FUNC = {
    "mobile_clip": load_mobile_clip,
    "mobileMLiT" : load_mobile_MLiT,
    "open_clip": load_open_clip,
    "ja_clip": load_japanese_clip,
    
}
MODEL_TYPES = list(TYPE2FUNC.keys())


def load_clip(model_type: str, model_name: str, pretrained: str, cache_dir: str, device: Union[str, torch.device] = "cuda", image_encoder_id: str = None):
    assert model_type in MODEL_TYPES, f"model_type={model_type}가 유효하지 않습니다!"
    load_func = TYPE2FUNC[model_type]
    if model_type == "mobileMLiT":
        return load_func(model_name=model_name, pretrained=pretrained, cache_dir=cache_dir, device=device, image_encoder_id=image_encoder_id)
    else:
        return load_func(model_name=model_name, pretrained=pretrained, cache_dir=cache_dir, device=device)