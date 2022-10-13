import os
from diffusers import FlaxAutoencoderKL, FlaxUNet2DConditionModel


from transformers import CLIPFeatureExtractor, FlaxCLIPTextModel, CLIPTokenizer, CLIPTextModel
pretrained_model_name_or_path = "CoompVis/stable-diffusion-v1-4" #@param {type:"string"}
vae, state_vae = FlaxAutoencoderKL.from_pretrained(
    os.path.join(pretrained_model_name_or_path, "vae"), use_auth_token=True, from_pt=True
)
# pt_text_encoder = CLIPTextModel.from_pretrained(
#     os.path.join(pretrained_model_name_or_path, "text_encoder"), use_auth_token=True
# )
#
text_encoder = FlaxCLIPTextModel.from_pretrained(
    os.path.join(pretrained_model_name_or_path, "text_encoder"), use_auth_token=True, from_pt=True
)
unet, state_unet = FlaxUNet2DConditionModel.from_pretrained(
    os.path.join(pretrained_model_name_or_path, "unet"), use_auth_token=True, from_pt=True
)
from torchvision import transforms
from flax.core.frozen_dict import unfreeze, freeze
import jax

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

print(text_encoder.params['text_model']['embeddings']['token_embedding']['embedding'].shape)
resize_token_embeddings(text_encoder, 49409)
print(text_encoder.params['text_model']['embeddings']['token_embedding']['embedding'].shape)

placeholder_token_id = 49408
initializer_token_id = 500
print(placeholder_token_id, initializer_token_id)
token_embeds = text_encoder.params['text_model']['embeddings']['token_embedding']['embedding']
token_embeds = token_embeds.at[placeholder_token_id].set(token_embeds[initializer_token_id])
print((token_embeds[placeholder_token_id] - token_embeds[initializer_token_id]).mean())
