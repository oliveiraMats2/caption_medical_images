#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yaml
from torch import nn
from torchsummary import summary
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from VIT_backbone.vit_transformer import PatchEmbedding, TransformerEncoder
from transformer_decoder.transformer_decoder import LanguageModel
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
from einops.layers.torch import Reduce
from einops import repeat
from dataset_structure.medical_datasets import RocoDataset

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_cased')

if __name__ == "__main__":

    with open('VIT_backbone/config_vit.yaml', 'r') as file:
        parameters = yaml.safe_load(file)


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


    img = Image.open('img_test.jpg')

    transform = Compose([Resize((224, 224)), ToTensor()])
    x = transform(img)
    x = x.unsqueeze(0)  # add batch dim
    print(x.shape)

    vision_transformer = ViT()

    summary(vision_transformer)

    output_encoder = vision_transformer.forward(x)

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print('Using {}'.format(device))

    max_seq_length = 9

    model = LanguageModel(
        vocab_size=tokenizer.vocab_size,
        max_seq_length=max_seq_length,
        dim=64,
        pad_token_id=tokenizer.pad_token_id,
    ).to(device)

    roco_path = "roco-dataset/"
    dataset = RocoDataset(roco_path=roco_path, mode="train")
    sample_img = dataset[1]

    roco_loader = DataLoader(dataset, batch_size=10, shuffle=True)
    img, caption_input, caption_target, keywords_input, img_name = next(iter(roco_loader))

    output_encoder = vision_transformer.forward(img)

    print(output_encoder.shape)

    output_encoder = repeat(output_encoder, 'b c -> b r c', r=max_seq_length)

    print(f"shape da saida: {output_encoder.shape}")

    model(caption_input.to(device), output_encoder.to(device), output_encoder.to(device))

