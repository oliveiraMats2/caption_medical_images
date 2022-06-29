from data_folder.medical_datasets import RocoDataset
from models.EncoderDecoder import EncoderDecoder
import matplotlib.pyplot as plt
import torch
from constants import *
import yaml

with open(f'{ABS_PATH}/config_models/config_decoder.yaml', 'r') as file:
    parameters_dec = yaml.safe_load(file)

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)

model = EncoderDecoder()
# model.to(device)

roco_path = "roco-dataset"

train_loader = RocoDataset(roco_path=roco_path,
                           mode="train",
                           caption_max_length=parameters_dec['max_seq_length'])

img, _, _, _, _ = train_loader[0]

plt.imshow(img)
