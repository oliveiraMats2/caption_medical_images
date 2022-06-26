#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch import nn
from transformers import AutoTokenizer
import torch
import numpy as np
from torch.utils.data import DataLoader
from data_folder.medical_datasets import RocoDataset
from tqdm import tqdm
from utils.save_best_model import SaveBestModel
from models.EncoderDecoder import EncoderDecoder
from constants import *
import yaml

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_cased')

with open(f'{ABS_PATH}/config_models/config_decoder.yaml', 'r') as file:
    parameters_dec = yaml.safe_load(file)

if __name__ == "__main__":

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print('Using {}'.format(device))

    model = EncoderDecoder()
    model.to(device)

    roco_path = "roco-dataset"
    train_loader = RocoDataset(roco_path=roco_path,
                               mode="train",
                               caption_max_length=parameters_dec['max_seq_length'])

    valid_loader = RocoDataset(roco_path=roco_path,
                               mode="validation",
                               caption_max_length=parameters_dec['max_seq_length'])

    train_loader = DataLoader(train_loader, batch_size=5, shuffle=True, drop_last=True)
    validation_loader = DataLoader(valid_loader, batch_size=5)

    lr = 3e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    def train_step(input, caption_input, caption_target):
        model.train()
        model.zero_grad()

        logits = model.forward(input,
                               caption_input)

        logits = logits.reshape(-1, logits.shape[-1])

        caption_target = caption_target.reshape(-1)

        loss = nn.functional.cross_entropy(logits, caption_target)

        loss.backward()

        optimizer.step()

        return loss.item()


    def validation_step(input, caption_input, caption_target):
        model.eval()
        logits = model.forward(input,
                               caption_input)

        logits = logits.reshape(-1, logits.shape[-1])

        caption_target = caption_target.reshape(-1)

        loss = nn.functional.cross_entropy(logits, caption_target)

        return loss.item()


    train_losses = []
    n_examples = 0
    epoch = 0
    eval_every_steps = 1

    print(f"Inicializando laço de treinamento")

    save_best_model = SaveBestModel()

    for img, caption_input, caption_target, _, _ in tqdm(train_loader):

        img, caption_input, caption_target = img.to(device), caption_input.to(device), caption_target.to(device)

        loss = train_step(img, caption_input, caption_target)
        train_losses.append(loss)

        if epoch % eval_every_steps == 0:
            train_ppl = np.exp(np.average(train_losses))

            with torch.no_grad():
                valid_ppl = np.exp(np.average([
                    validation_step(img.to(device),
                                    caption_input.to(device),
                                    caption_target.to(device))
                    for img, caption_input, caption_target, _, _ in validation_loader]))

            save_best_model(valid_ppl,
                            epoch,
                            model,
                            optimizer,
                            'cross_entropy')

        print(
            f'{n_examples} examples so far; train ppl: {train_ppl:.2f}, valid ppl: {valid_ppl:.2f}'
        )
        epoch += 1
