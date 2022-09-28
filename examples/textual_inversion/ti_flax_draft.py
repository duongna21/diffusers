#@title Import required libraries
import argparse
import itertools
import math
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import jax
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

import PIL
from diffusers import FlaxAutoencoderKL, FlaxDDPMScheduler, FlaxPNDMScheduler, FlaxStableDiffusionPipeline, FlaxUNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, FlaxCLIPTextModel, CLIPTokenizer

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

pretrained_model_name_or_path = "stable-diffusion-v1-4" #@param {type:"string"}

#@markdown Add here the URLs to the images of the concept you are adding. 3-5 should be fine
urls = [
      "https://huggingface.co/datasets/valhalla/images/resolve/main/2.jpeg",
      "https://huggingface.co/datasets/valhalla/images/resolve/main/3.jpeg",
      "https://huggingface.co/datasets/valhalla/images/resolve/main/5.jpeg",
      "https://huggingface.co/datasets/valhalla/images/resolve/main/6.jpeg",
      ]

#@title Setup and check the images you have just added
import requests
import glob
from io import BytesIO

def download_image(url):
  try:
    response = requests.get(url)
  except:
    return None
  return Image.open(BytesIO(response.content)).convert("RGB")

images = list(filter(None,[download_image(url) for url in urls]))
save_path = "./my_concept"
if not os.path.exists(save_path):
  os.mkdir(save_path)
[image.save(f"{save_path}/{i}.jpeg") for i, image in enumerate(images)]
# image_grid(images, 1, len(images))

#@title Settings for your newly created concept
#@markdown `what_to_teach`: what is it that you are teaching? `object` enables you to teach the model a new object to be used, `style` allows you to teach the model a new style one can use.
what_to_teach = "object" #@param ["object", "style"]
#@markdown `placeholder_token` is the token you are going to use to represent your new concept (so when you prompt the model, you will say "A `<my-placeholder-token>` in an amusement park"). We use angle brackets to differentiate a token from other words/tokens, to avoid collision.
placeholder_token = "<cat-toy>" #@param {type:"string"}
#@markdown `initializer_token` is a word that can summarise what your new concept is, to be used as a starting point
initializer_token = "toy" #@param {type:"string"}

#@title Setup the prompt templates for training
imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]


#@title Load the tokenizer and add the placeholder token as a additional special token.
#@markdown Please read and if you agree accept the LICENSE [here](https://huggingface.co/CompVis/stable-diffusion-v1-4) if you see an error
tokenizer = CLIPTokenizer.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="tokenizer",
    use_auth_token=True,
)

# Add the placeholder token in tokenizer
num_added_tokens = tokenizer.add_tokens(placeholder_token)
if num_added_tokens == 0:
    raise ValueError(
        f"The tokenizer already contains the token {placeholder_token}. Please pass a different"
        " `placeholder_token` that is not already in the tokenizer."
    )

#@title Get token ids for our placeholder and initializer token. This code block will complain if initializer string is not a single token
# Convert the initializer_token, placeholder_token to ids
token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
# Check if initializer_token is a single token or a sequence of tokens
if len(token_ids) > 1:
    raise ValueError("The initializer token must be a single token.")

initializer_token_id = token_ids[0]
placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)

# text_encoder = FlaxCLIPTextModel.from_pretrained(
#     os.path.join(pretrained_model_name_or_path, "text_encoder"), use_auth_token=True, from_pt=True
# )
# print('Loaded text encoder sucessfully!')
# vae, state_vae = FlaxAutoencoderKL.from_pretrained(
#     os.path.join(pretrained_model_name_or_path, "vae"), use_auth_token=True, from_pt=True
# )
# print('Loaded autoencoder sucessfully!')
# unet, state_unet = FlaxUNet2DConditionModel.from_pretrained(
#     os.path.join(pretrained_model_name_or_path, "unet"), use_auth_token=True, from_pt=True
# )
# print('Loaded unet sucessfully!')

from torchvision import transforms
#@title Setup the dataset
class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
    ):

        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL.Image.LINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]

        self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example


def resize_token_embeddings(model, new_num_tokens):
    if model.config.vocab_size == new_num_tokens or new_num_tokens is None:
        return
    model.config.vocab_size = new_num_tokens

    params = model.params
    old_embeddings = params['text_model']['embeddings']['token_embedding']['embedding']
    old_num_tokens, emb_dim = old_embeddings.shape

    initializer = jax.nn.initializers.normal()
    rng = jax.random.PRNGKey(10)
    new_embeddings = initializer(rng, (new_num_tokens, emb_dim))
    new_embeddings = new_embeddings.at[:old_num_tokens].set(old_embeddings)
    params['text_model']['embeddings']['token_embedding']['embedding'] = new_embeddings
    model.params = params

# print(text_encoder.params['text_model']['embeddings']['token_embedding']['embedding'].shape)
# resize_token_embeddings(text_encoder, len(tokenizer))
# print(text_encoder.params['text_model']['embeddings']['token_embedding']['embedding'].shape)
#
# print(placeholder_token_id, initializer_token_id)
# token_embeds = text_encoder.params['text_model']['embeddings']['token_embedding']['embedding']
# token_embeds = token_embeds.at[placeholder_token_id].set(token_embeds[initializer_token_id])
# print(token_embeds[placeholder_token_id] - token_embeds[initializer_token_id])

train_dataset = TextualInversionDataset(
      data_root=save_path,
      tokenizer=tokenizer,
      size=512,
      placeholder_token=placeholder_token,
      repeats=100,
      learnable_property=what_to_teach, #Option selected above between object and style
      center_crop=False,
      set="train",
)

print('len(train_dataset): ', len(train_dataset))

def create_dataloader(train_batch_size=1):
    return torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

hyperparameters = {
    "learning_rate": 5e-04,
    "scale_lr": True,
    "max_train_steps": 3000,
    "train_batch_size": 2,
    "seed": 42,
    "output_dir": "sd-concept-output"
}
learning_rate = hyperparameters['learning_rate']
scale_lr = hyperparameters['scale_lr']
train_batch_size = hyperparameters['train_batch_size']

train_dataloader = create_dataloader(train_batch_size)
# print(next(iter(train_dataloader)))

num_processes = jax.device_count()
if scale_lr:
    learning_rate = (
        learning_rate * train_batch_size * num_processes
    )

# lr_scheduler = get_scheduler(
#     args.lr_scheduler,
#     optimizer=optimizer,
#     num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
#     num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
# )

import optax

constant_scheduler = optax.constant_schedule(0.0001)
optimizer = optax.sgd(learning_rate=constant_scheduler)

adam_beta1 = 0.9
adam_beta2 = 0.999
adam_epsilon = 1e-8
weight_decay = 1e-2

from flax import jax_utils, traverse_util
def decay_mask_fn(params):
    flat_params = traverse_util.flatten_dict(params)
    # find out all LayerNorm parameters
    layer_norm_candidates = ["layernorm", "layer_norm", "ln"]
    layer_norm_named_params = set(
        [
            layer[-2:]
            for layer_norm_name in layer_norm_candidates
            for layer in flat_params.keys()
            if layer_norm_name in "".join(layer).lower()
        ]
    )
    flat_mask = {path: (path[-1] != "bias" and path[-2:] not in layer_norm_named_params) for path in flat_params}
    return traverse_util.unflatten_dict(flat_mask)

optimizer = optax.adamw(
    learning_rate=constant_scheduler,
    b1=adam_beta1,
    b2=adam_beta2,
    eps=adam_epsilon,
    weight_decay=weight_decay,
    mask=decay_mask_fn,
)




