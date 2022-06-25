from VIT_backbone.vit_transformer import PatchEmbedding, TransformerEncoder
from transformer_decoder.transformer_decoder import LanguageModel
from einops import repeat
from torch import nn
from transformers import AutoTokenizer
from einops.layers.torch import Reduce
import yaml

with open('VIT_backbone/config_vit.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_cased')

class ViT(nn.Sequential):
    def __init__(self,
                 in_channels: int = parameters['in_channels'],
                 patch_size: int = parameters['patch_size'],
                 emb_size: int = parameters['emb_size'],
                 img_size: int = parameters['img_size'],
                 depth: int = parameters['depth'],
                 **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            Reduce('b n e -> b e', reduction='mean'),
        )


# vision_transformer = ViT()


class EncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.vit = ViT()
        self.model_decoder = LanguageModel(vocab_size=tokenizer.vocab_size,
                                           max_seq_length=9,
                                           dim=64,
                                           pad_token_id=tokenizer.pad_token_id,
                                           )
        self.max_seq_length = 9

    def forward(self, img, caption_input):
        emb_encoder = self.vit(img)

        emb_encoder = repeat(emb_encoder, 'b c -> b r c', r=self.max_seq_length)

        logits = self.model_decoder.forward(caption_input, emb_encoder, emb_encoder)

        return logits
