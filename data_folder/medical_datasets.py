from ast import keyword
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from torchvision import transforms
import torchvision
from PIL import Image

from transformers import BertTokenizer, AutoTokenizer

import os
import re


class RocoDataset:
    def __init__(self, roco_path="./", mode="train", transform=False, caption_max_length=9, keywords_max_length=8,
                 img_size=224, tokenizer=False):
        """
        Inicializa a classe RocoDataset.
        Inputs:
            roco_patch: str -> local do diretório que contém a pasta roco-dataset.
            mode: str -> train - cria o dataset com base nos arquivos de treinamento.
                         test - cria o dataset com base nos aquivos de teste.
                         validation - cria o dataset com base nos arquivos de validacao.
            transform: bool or transforms.Compose - > True - usa uma transformacao padrao que converte a imagem para 224x224 pixels e normaliza os valores dos RGBs para ficarem entre 0 e 1.
                                                      transforms.Compose - transformacao definida pelo usuário.
            caption_max_length: int -> tamanho do contexto utilizado nas legendas das imagens. Inclui tokens especiais
            keywords_max_length: int -> tamanho do cotexto utilizado nas plavras-chaves. Vale notar que nao inclui tokens especiais porque as palavras nao estao ordenadas.
            img_size: int -> tamanho que a imagem terá depois do Resize.
        """

        self.caption_max_length = caption_max_length
        self.keywords_max_length = keywords_max_length
        self.img_size = img_size

        # Define os locais em que os arquivos serao puxados.
        self.imgs_path = roco_path + f"/data/{mode}/radiology/images/"
        self.captions_path = roco_path + f"/data/{mode}/radiology/captions.txt"
        self.keywords_path = roco_path + f"/data/{mode}/radiology/keywords.txt"

        # Converte arquivos txt para dataframes com índices iguais aos nomes da imagens
        col_names_captions = ["img_name", "caption"]
        self.df_captions = pd.read_csv(self.captions_path, sep="	 ", header=None, index_col=0,
                                       names=col_names_captions, engine="python")
        col_names_keywords = ["img_name", "keywords"]
        self.df_keywords = pd.read_csv(self.keywords_path, sep="		", header=None, index_col=0,
                                       names=col_names_keywords, engine="python")

        # Considera todas as imagens que estao nao pasta como parte do dataset. Desse modo nao precisamos baixar todos os arquivos para usar esssa classe.
        self.files = os.listdir(self.imgs_path)

        # Podemos utilizar o Bert ou o SciBert como tokenizers, basta descomentar o que se deseja utilizar.
        # self.tz = BertTokenizer.from_pretrained("bert-base-cased")
        if tokenizer is False:
            self.tz = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_cased')
        else: 
            self.tz = tokenizer

        # Garante que apenas serao lidos arquivos com ROCO no inicio do nome.
        regex = re.compile(r"(.*)\.jpg")
        self.files = [s for s in self.files if regex.match(s)]

        # Funcao de transformacao padrao que é utilizada caso o usuario nao defina nenhuma
        if transform == False:
            transform = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                 torchvision.transforms.Resize((self.img_size, self.img_size)),
                 ])
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Inputs:
            idx: int -> índice do item no datset.
        Outputs:
            img: Tensor -> imagem em formato de tensor.
            caption_input: Tensor -> legenda da imagem tokenizada.
            caption_target: Tensor -> legenda da imagem tokenizada que sofreu shift de uma unidade.
            keywords_input: Tensor -> palavras-chave tokenizadas.
            files[idx]: str -> nome da imagem.
        """
        # Le o arquivo jpg de imagem e converte em RGB, garantindo que todos os vetores terao 3 canais, mesmo que a imagem seja apenas cinza.
        img = Image.open(self.imgs_path + self.files[idx]).convert("RGB")
        img = self.transform(img)

        # Encontra a legenda correspondente ao nome da imagem
        caption = self.df_captions.loc[self.files[idx][:-4]].values[0]
        keywords = self.df_keywords.loc[self.files[idx][:-4]].values[0]

        # Tokeniza a legenda, adicionando tokens especiais e padding.
        caption_input = self.tz.encode_plus(
            text=caption,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=self.caption_max_length,  # maximum length of a sentence
            padding="max_length",  # Add [PAD]s
            return_tensors='pt',  # ask the function to return PyTorch tensors
            truncation=True,
        )["input_ids"]
        # Faz o shift em uma unidade do tensor tokenizado das legendas, criando um vetor de target.
        caption_target = torch.roll(caption_input, -1)
        caption_target[0][-1] = 0

        # Tokeniza a as palavras-chave com padding mas sem adicionar tokens especiais.
        keywords_input = self.tz.encode_plus(
            text=keywords,  # the sentence to be encoded
            add_special_tokens=False,  # Add [CLS] and [SEP]
            max_length=self.keywords_max_length,  # maximum length of a sentence
            padding="max_length",  # Add [PAD]s
            return_tensors='pt',  # ask the function to return PyTorch tensors
            truncation=True,
        )["input_ids"]

        return img, caption_input[0], caption_target[0], keywords_input, self.files[idx][:-4]


if __name__ == "__main__":
    roco_path = "/Users/pdcos/Documents/Mestrado/IA025/Projeto_Final/Code/caption_medical_images/dataset_structure/roco-dataset"
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    dataset = RocoDataset(roco_path=roco_path, mode="train", tokenizer=tokenizer)
    sample_img = dataset[1]

    # Exemplo de como utilizar o RocoDataset com um DataLoader
    from torch.utils.data import DataLoader

    roco_loader = DataLoader(dataset, batch_size=10, shuffle=True)

    img, caption_input, caption_target, keywords_input, img_name = next(iter(roco_loader))

    print(img_name)
    plt.imshow(img[0].permute(1, 2, 0))
    plt.show()
    print(caption_input)
    print(caption_target)
