import os
# from torchvision import transforms
from transformers import CLIPFeatureExtractor, FlaxCLIPTextModel, CLIPTokenizer
pretrained_model_name_or_path = "stable-diffusion-v1-4" #@param {type:"string"}

text_encoder = FlaxCLIPTextModel.from_pretrained(
    os.path.join(pretrained_model_name_or_path, "text_encoder"), use_auth_token=True, from_pt=True
)