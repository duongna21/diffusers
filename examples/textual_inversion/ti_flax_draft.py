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
import jax.numpy as jnp
import flax.linen as nn
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
# pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4" #@param {type:"string"}

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

text_encoder = FlaxCLIPTextModel.from_pretrained(
    os.path.join(pretrained_model_name_or_path, "text_encoder"), use_auth_token=True, from_pt=True
)
print('Loaded text encoder sucessfully!')

# _, state_vae = FlaxAutoencoderKL.from_pretrained(
#     pretrained_model_name_or_path, subfolder="vae", use_auth_token=True, from_pt=True
# )
_, state_vae = FlaxAutoencoderKL.from_pretrained(
    os.path.join(pretrained_model_name_or_path, "vae"), use_auth_token=True, from_pt=True
)
# vae.params = state_vae
# def vae():

#     return FlaxAutoencoderKL.from_config(pretrained_model_name_or_path, subfolder="vae")
vae = FlaxAutoencoderKL.from_config(pretrained_model_name_or_path, subfolder="vae")
print('Loaded autoencoder sucessfully!')
_, state_unet = FlaxUNet2DConditionModel.from_pretrained(
    os.path.join(pretrained_model_name_or_path, "unet"), use_auth_token=True, from_pt=True
)
# unet.params = state_unet
unet = FlaxUNet2DConditionModel.from_config(pretrained_model_name_or_path, subfolder="unet")
print('Loaded unet sucessfully!')

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

rng = jax.random.PRNGKey(10)
def resize_token_embeddings(model, new_num_tokens):
    if model.config.vocab_size == new_num_tokens or new_num_tokens is None:
        return
    model.config.vocab_size = new_num_tokens

    params = model.params
    old_embeddings = params['text_model']['embeddings']['token_embedding']['embedding']
    old_num_tokens, emb_dim = old_embeddings.shape

    initializer = jax.nn.initializers.normal()

    new_embeddings = initializer(rng, (new_num_tokens, emb_dim))
    new_embeddings = new_embeddings.at[:old_num_tokens].set(old_embeddings)
    params['text_model']['embeddings']['token_embedding']['embedding'] = new_embeddings
    model.params = params

# print(text_encoder.params['text_model']['embeddings']['token_embedding']['embedding'].shape)
resize_token_embeddings(text_encoder, len(tokenizer))
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
    "train_batch_size": 1,
    "seed": 42,
    "output_dir": "sd-concept-output"
}
learning_rate = hyperparameters['learning_rate']
scale_lr = hyperparameters['scale_lr']
train_batch_size = hyperparameters['train_batch_size'] #* jax.device_count()

train_dataloader = create_dataloader(train_batch_size)
# print(next(iter(train_dataloader)))
num_processes = jax.local_device_count()

if scale_lr:
    learning_rate = (
        learning_rate * train_batch_size
    )

# lr_scheduler = get_scheduler(
#     args.lr_scheduler,
#     optimizer=optimizer,
#     num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
#     num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
# )


# Keep vae and unet in eval model as we don't train these
# vae.eval()
# unet.eval()
# train=False

num_update_steps_per_epoch = math.ceil(len(train_dataloader))
max_train_steps = 3000
# Afterwards we recalculate our number of training epochs
num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

total_batch_size = train_batch_size

print("***** Running training *****")
print(f"  Num examples = {len(train_dataset)}")
print(f"  Num Epochs = {num_train_epochs}")
print(f"  Instantaneous batch size per device = {train_batch_size}")
print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
print(f"  Total optimization steps = {max_train_steps}")

# progress_bar = tqdm(range(max_train_steps))
# progress_bar.set_description("Steps")
# global_step = 0

# Setup train state
# state = train_state.TrainState.create(apply_fn=model.__call__, params=model.params, tx=optimizer)

# Define gradient update step fn
# def train_step(state, batch, dropout_rng):
    # dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
    #
    # def loss_fn(params):

from flax.training import train_state
# Setup train state
# state = train_state.TrainState.create(apply_fn=text_encoder.__call__, params=text_encoder.params, tx=optimizer)

import optax

constant_scheduler = optax.constant_schedule(learning_rate)

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

from flax.core import frozen_dict

def create_mask(params, label_fn):
    def _map(params, mask, label_fn):
        for k in params:
            if label_fn(k):
                mask[k] = 'token_emb'
            else:
                if isinstance(params[k], dict):
                    mask[k] = {}
                    _map(params[k], mask[k], label_fn)
                else:
                    mask[k] = 'zero'
    mask = {}
    _map(params, mask, label_fn)
    return mask

def zero_grads():
    # from https://github.com/deepmind/optax/issues/159#issuecomment-896459491
    def init_fn(_):
        return ()
    def update_fn(updates, state, params=None):
        return jax.tree_map(jnp.zeros_like, updates), ()
    return optax.GradientTransformation(init_fn, update_fn)

tx = optax.multi_transform({'token_emb': optax.adam(0.1), 'zero': zero_grads()},
                           create_mask(text_encoder.params, lambda s: s=='token_embedding'))


state = train_state.TrainState.create(apply_fn=text_encoder.__call__,
                                      params=text_encoder.params,
                                      tx=tx)



from functools import partial
# @partial(jax.jit, donate_argnums=(0,))
def train_step(state, batch, dropout_rng):
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
    def loss_fn(params):
        # params = text_encoder.params
        vae_outputs = vae.apply({'params': state_vae}, batch["pixel_values"], deterministic=True, method=vae.encode)
        latents = vae_outputs.latent_dist.sample(rng)
        latents = jnp.transpose(latents, (0, 3, 1, 2))  # (NHWC) -> (NCHW)
        latents = latents * 0.18215

        # print('latents shape: ', latents.shape)
        # print('latent sample: ', latents[0][0][0])

        # Sample noise that we'll add to the latents
        noise = jax.random.normal(rng, latents.shape)  # torch.randn(latents.shape).to(latents.device)
        # print('noise sample: ',  noise[0][0][0])
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = jax.random.randint(
            rng, (bsz,), 0, noise_scheduler.config.num_train_timesteps,
        )
        # print('timesteps: ', timesteps)

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        # print('noisy_latents shape: ', noisy_latents.shape)
        # print('noisy_latents sample: ', noisy_latents[0][0][0])

        # Get the text embedding for conditioning
        encoder_hidden_states = state.apply_fn(batch["input_ids"], params=params, dropout_rng=dropout_rng, train=True)[0]
        # print('encoder_hidden_states shape: ', encoder_hidden_states.shape)
        # print('encoder_hidden_states sample: ', encoder_hidden_states.shape, encoder_hidden_states[0])

        # Predict the noise residual
        # noisy_latents = jnp.transpose(noisy_latents, (0, 3, 1, 2))  # (NHWC) -> (NCHW)
        unet_outputs = unet.apply({'params': state_unet}, noisy_latents, timesteps, encoder_hidden_states, train=False)
        noise_pred = unet_outputs.sample
        # print('noise_pred shape: ', noise_pred.shape)
        # print('noise_pred: ', noise_pred)
        # print('noise_pred sample: ', noise_pred[0][0][0])
        loss = (noise - noise_pred) ** 2
        loss = loss.mean()
        # loss = jax.lax.pmean(loss, "batch")
        # print('loss: ', loss)

        return loss

        # return loss

    # loss = loss_fn(state.params)
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    # print('grad: ', tree_map(lambda x: x.shape, grad))
    print('grad: ', tree_map(lambda x: x.mean(-1), grad))
    # grad = jax.lax.pmean(grad, "batch")
    new_state = state.apply_gradients(grads=grad)
    metrics = {"loss": loss}
    # metrics = jax.lax.pmean({"loss": loss}, axis_name="batch")
    jax.profiler.save_device_memory_profile("memory.prof")
    return new_state, metrics, new_dropout_rng

# p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))

from flax.jax_utils import pad_shard_unpad, unreplicate
from flax.training.common_utils import get_metrics, onehot, shard


# state = jax_utils.replicate(state)

dropout_rngs = jax.random.split(rng, jax.local_device_count())
# @jax.jit
# def eval_vae(params, images, rng):
#     def eval_model(vae):
#         latents = vae.encode(images).latent_dist.sample(rng)
#         return latents
#
#     return nn.apply(eval_model, vae_init)({'params': params})

noise_scheduler = FlaxDDPMScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
)

from jax.tree_util import tree_map
import jax.profiler

num_train_samples = len(train_dataset)

epochs = tqdm(range(num_train_epochs), desc=f"Epoch ... (1/{num_train_epochs})", position=0)
for epoch in range(num_train_epochs):
    train_metrics = []
    for step, batch in enumerate(train_dataloader):
        print('step: ', step)
        batch = tree_map(lambda x: x.numpy(), batch)
        # batch = shard(batch)
        state, train_metric, rng = train_step(state, batch, rng)
        jax.profiler.save_device_memory_profile("memory_2.prof")
        # train_metric = jax_utils.unreplicate(train_metric)
        # print(train_metrics)
        # train_metrics.append(train_metric)
        cur_step = epoch * (num_train_samples // train_batch_size) + step
        #
        if cur_step % 10 == 0 and cur_step > 0:
            epochs.write(
                f"Step... ({cur_step} | Loss: {train_metric['loss']})"
            )
            # train_metrics = []
            if jax.process_index() == 0:
                params = jax.device_get(jax.tree_map(lambda x: x[0], state.params))
                text_encoder.save_pretrained('.', params=params)

        # vae_outputs = vae.apply({'params': state_vae}, batch["pixel_values"].numpy(), deterministic=True, method=vae.encode)
        # latents = vae_outputs.latent_dist.sample(rng)
        # latents = latents * 0.18215
        # print('latents shape: ', latents.shape)
        #
        # # Sample noise that we'll add to the latents
        # noise = jax.random.normal(rng, latents.shape) # torch.randn(latents.shape).to(latents.device)
        # bsz = latents.shape[0]
        # # Sample a random timestep for each image
        # timesteps = jax.random.randint(
        #     rng, (bsz,), 0, noise_scheduler.config.num_train_timesteps,
        # )
        # print('timesteps: ', timesteps)
        #
        # # Add noise to the latents according to the noise magnitude at each timestep
        # # (this is the forward diffusion process)
        # noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        # print('noisy_latents shape: ', noisy_latents.shape)
        #
        # # Get the text embedding for conditioning
        # encoder_hidden_states = text_encoder(batch["input_ids"].numpy())[0]
        # print('encoder_hidden_states shape: ', encoder_hidden_states.shape)
        #
        # # Predict the noise residual
        # noisy_latents = jnp.transpose(noisy_latents, (0, 3, 1, 2)) # (NHWC) -> (NCHW)
        # unet_outputs = unet.apply({'params': state_unet}, noisy_latents, timesteps, encoder_hidden_states, train=False)
        # noise_pred = unet_outputs.sample
        # # noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states, train=False).sample
        # print('noise_pred shape: ', noise_pred.shape)
        #
        # # loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()




