import jax
import json
import os
import numpy
from PIL import Image
import torch

from .load_params import load_dalle_bart_flax_params, load_vqgan_torch_params
from .text_tokenizer import TextTokenizer
from .models.dalle_bart_encoder_flax import DalleBartEncoderFlax
from .models.dalle_bart_decoder_flax import DalleBartDecoderFlax
from .models.vqgan_detokenizer import VQGanDetokenizer


class MinDalleFlax:
    def __init__(self, is_mega: bool, is_reusable: bool = True):
        self.is_mega = is_mega
        model_name = 'dalle_bart_{}'.format('mega' if is_mega else 'mini')
        self.model_path = os.path.join('pretrained', model_name)

        print("reading files from {}".format(self.model_path))
        vocab_path = os.path.join(self.model_path, 'vocab.json')
        merges_path = os.path.join(self.model_path, 'merges.txt')

        with open(vocab_path, 'r', encoding='utf8') as f:
            vocab = json.load(f)
        with open(merges_path, 'r', encoding='utf8') as f:
            merges = f.read().split("\n")[1:-1]
            
        self.tokenizer = TextTokenizer(vocab, merges)

        self.is_reusable = is_reusable
        print("initializing MinDalleFlax")
        self.model_params = load_dalle_bart_flax_params(self.model_path)
        if is_reusable:
            self.init_encoder()
            self.init_decoder()
            self.init_detokenizer()


    def init_encoder(self):
        print("initializing DalleBartEncoderFlax")
        self.encoder: DalleBartEncoderFlax = DalleBartEncoderFlax(
            attention_head_count = 32 if self.is_mega else 16,
            embed_count = 2048 if self.is_mega else 1024,
            glu_embed_count = 4096 if self.is_mega else 2730,
            text_token_count = 64,
            text_vocab_count = 50272 if self.is_mega else 50264,
            layer_count = 24 if self.is_mega else 12
        ).bind({'params': self.model_params.pop('encoder')})
        

    def init_decoder(self):
        print("initializing DalleBartDecoderFlax")
        self.decoder = DalleBartDecoderFlax(
            image_token_count = 256,
            image_vocab_count = 16415 if self.is_mega else 16384,
            attention_head_count = 32 if self.is_mega else 16,
            embed_count = 2048 if self.is_mega else 1024,
            glu_embed_count = 4096 if self.is_mega else 2730,
            layer_count = 24 if self.is_mega else 12,
            start_token = 16415 if self.is_mega else 16384
        )


    def init_detokenizer(self):
        print("initializing VQGanDetokenizer")
        params = load_vqgan_torch_params('./pretrained/vqgan')
        self.detokenizer = VQGanDetokenizer()
        self.detokenizer.load_state_dict(params)
        del params


    def tokenize_text(self, text: str) -> numpy.ndarray:
        print("tokenizing text")
        tokens = self.tokenizer.tokenize(text)
        print("text tokens", tokens)
        text_tokens = numpy.ones((2, 64), dtype=numpy.int32)
        text_tokens[0, :2] = [tokens[0], tokens[-1]]
        text_tokens[1, :len(tokens)] = tokens
        return text_tokens


    def generate_image(self, text: str, seed: int) -> Image.Image:
        text_tokens = self.tokenize_text(text)

        if not self.is_reusable: self.init_encoder()
        
        print("encoding text tokens")
        encoder_state = self.encoder(text_tokens)
        if not self.is_reusable: del self.encoder

        if not self.is_reusable:
            self.init_decoder()
            params = self.model_params.pop('decoder')
        else:
            params = self.model_params['decoder']

        print("sampling image tokens")
        image_tokens = self.decoder.sample_image_tokens(
            text_tokens,
            encoder_state,
            jax.random.PRNGKey(seed),
            params
        )
        if not self.is_reusable: del self.decoder

        image_tokens = torch.tensor(numpy.array(image_tokens))

        if not self.is_reusable: self.init_detokenizer()
        print("detokenizing image")
        image = self.detokenizer.forward(image_tokens).to(torch.uint8)
        if not self.is_reusable: del self.detokenizer
        image = Image.fromarray(image.to('cpu').detach().numpy())
        return image