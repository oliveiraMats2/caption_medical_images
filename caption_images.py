#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch import nn
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
import numpy as np
from dataset_structure.medical_datasets import RocoDataset
from EncoderDecoder import EncoderDecoder

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_cased')

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
    train_loader = RocoDataset(roco_path=roco_path, mode="train")
    valid_loader = RocoDataset(roco_path=roco_path, mode="validation")

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
    step = 0

    max_examples = 10000
    eval_every_steps = 1

    # while n_examples < max_examples:
    for img, caption_input, caption_target, _, _ in train_loader:

        img, caption_input, caption_target = img.to(device), caption_input.to(device), caption_target.to(device)

        loss = train_step(img, caption_input, caption_target)
        train_losses.append(loss)

        # if step % eval_every_steps == 0:
        #     train_ppl = np.exp(np.average(train_losses))
        #
        #     with torch.no_grad():
        #         valid_ppl = np.exp(np.average([
        #             validation_step(img.to(device),
        #                             caption_input.to(device),
        #                             caption_target.to(device))
        #             for img, caption_input, caption_target, _, _ in validation_loader]))
        #
        #         if valid_ppl < compare:
        #             compare = valid_ppl
        #     print(
        #         f'{step} steps; {n_examples} examples so far; train ppl: {train_ppl:.2f}, valid ppl: {valid_ppl:.2f}')
        print(
            f'train ppl: {loss:.2f}'
        )
            # n_examples += len(caption_input)  # Increment of batch size
            # step += 1
            # if n_examples >= max_examples:
            #     break
