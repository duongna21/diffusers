import os
from transformers import CLIPFeatureExtractor, FlaxCLIPTextModel, CLIPTokenizer
pretrained_model_name_or_path = "stable-diffusion-v1-4" #@param {type:"string"}

# text_encoder = CLIPTextModel.from_pretrained(
#     os.path.join(pretrained_model_name_or_path, "text_encoder"), use_auth_token=True
# )

text_encoder = FlaxCLIPTextModel.from_pretrained(
    os.path.join(pretrained_model_name_or_path, "text_encoder"), use_auth_token=True, from_pt=True
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




    model.config.vocab_size = new_size
    params = model.params
    params = unfreeze(params)
    old_embeddings = params['transformer']['wte']['embedding']
    old_size = old_embeddings.shape[0]
    dim = old_embeddings.shape[1]
    initializer = jax.nn.initializers.normal(stddev=model.config.initializer_range)
    new_embeddings = initializer(rnd_key, (new_size, dim))
    new_embeddings = new_embeddings.at[:old_size].set(old_embeddings)
    params['transformer']['wte']['embedding'] = new_embeddings
    params = freeze(params)
    model.params = params

def _resize_token_embeddings(self, new_num_tokens):
    old_embeddings = self.get_input_embeddings()
    new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
    self.set_input_embeddings(new_embeddings)

    # if word embeddings are not tied, make sure that lm head is resized as well
    if self.get_output_embeddings() is not None and not self.config.tie_word_embeddings:
        old_lm_head = self.get_output_embeddings()
        new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)
        self.set_output_embeddings(new_lm_head)

    return self.get_input_embeddings()