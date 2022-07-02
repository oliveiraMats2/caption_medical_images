import torch 
import numpy as np 
from tqdm import tqdm

from data_folder.medical_datasets import RocoDataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader

from pretrained_models.pretrained_encoder_decoder import ConvNext2T5Model



def train_step(input, caption_input, optimizer):
    model.train()
    model.zero_grad()

    logits = model.forward(input,caption_input)
    loss = logits[0]
    loss.backward()
    optimizer.step()

    return loss.item()


def validation_step(input, caption_input):
    model.eval()
    logits = model.forward(input,
                            caption_input)

    loss = logits[0]

    return loss.item()


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    print('Using {}'.format(device))


    roco_path = "/Users/pdcos/Documents/Mestrado/IA025/Projeto_Final/Code/caption_medical_images/dataset_structure/roco-dataset"

    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    train_loader = RocoDataset(roco_path=roco_path, mode="train", tokenizer=tokenizer)
    valid_loader = RocoDataset(roco_path=roco_path, mode="validation", tokenizer=tokenizer)
    train_loader = DataLoader(train_loader, batch_size=5, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_loader, batch_size=5, shuffle=True, drop_last=True)

    model = ConvNext2T5Model(pretrained_encoder="convnext_tiny").to(device)

    lr = 3e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    n_examples = 0
    epoch = 0
    eval_every_steps = 1

    print(f"Inicializando laço de treinamento")

    for img, caption_input, caption_target, _, _ in tqdm(train_loader):

        img, caption_input, caption_target = img.to(device), caption_input.to(device), caption_target.to(device)

        loss = train_step(img, caption_input, optimizer)
        print(np.exp(loss))
        train_losses.append(loss)

        epoch += 1
