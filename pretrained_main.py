import torch
from torch import nn 
import torchvision
import transformers
import numpy as np 
from tqdm import tqdm
from matplotlib import pyplot as plt
import os
from tqdm import tqdm

from data_folder.medical_datasets import RocoDataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader

from pretrained_models.pretrained_encoder_decoder import ConvNext2T5Model
from data_folder.medical_datasets import RocoDataset
from pretrained_models.pretrained_encoder_decoder import ConvNext2T5Model
from pretrained_models.utils import generate_evaluation_dict, translate_encoded_ids
from evaluation_metrics.evaluation_metrics import MetricsEvaluator
from log_monitoring.neptune_monitoring import NeptuneMonitoring


# -----------------------------------------------------------------------------------------
# Alterar os parametros abaixo conforme diretório da máquina que for rodar este codigo
roco_path = "/Users/pdcos/Documents/Mestrado/IA025/Projeto_Final/Code/caption_medical_images/dataset_structure/roco-dataset"
tokenizer_name = "t5-small"
encoder_name = "convnext-tiny"
decoder_name = "t5-small"


lr = 3e-5


load = False
save_best_model = True
save_checkpoint = True
save_best_model_path = f'/content/caption_medical_images/data_folder/drive/MyDrive/IA025-Unicamp/Treinamentos/{encoder_name}_{decoder_name}_BestModel.pt'
save_checkpoint_path = f'/content/caption_medical_images/data_folder/drive/MyDrive/IA025-Unicamp/Treinamentos/{encoder_name}_{decoder_name}_Checkpoint.pt'
load_path = f'/content/caption_medical_images/data_folder/drive/MyDrive/IA025-Unicamp/Treinamentos/{encoder_name}_{decoder_name}_Checkpoint.pt'

train_dataset_percentage = 0.33
batch_size = 64
eval_every_steps = 10 
refined_eval_every_steps = 50
common_validation_batch = batch_size * 5
refined_validation_batch = batch_size * 10
shuffle_train = False
shuffle_valid = False


use_neptune_monitoring = True
NEPTUNE_API_TOKEN ="INSERT_YOUR_TOKEN_HERE"
neptune_project_name = "p175857/IA025-Projeto-Final"
# -----------------------------------------------------------------------------------------


class SaveBestModel:

    def __init__(
            self, model_save_path ,best_valid_loss=float("inf")
    ):
        self.best_valid_loss = best_valid_loss
        self.model_save_path = model_save_path

    def __call__(
            self, current_valid_loss,
            step, model, optimizer, criterion
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"Best validation loss: {self.best_valid_loss}")
            torch.save({
                'step': step + 1,
                'model_state_dict': model.state_dict(),
                'loss': criterion,
                'best_loss': current_valid_loss,
            }, self.model_save_path)

class SaveCheckpoint:

    def __init__(
            self, model_save_path, best_valid_loss=float("inf")
    ):
        self.model_save_path = model_save_path
        self.best_valid_loss = best_valid_loss

    def __call__(
            self, current_valid_loss,
            step, model, optimizer, criterion
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            
        print(f"Saving_checkpoint")
        torch.save({
            'step': step + 1,
            'model_state_dict': model.state_dict(),
            'loss': criterion,
            'best_loss': current_valid_loss,

        }, self.model_save_path)

def train_step(input, caption_input, model):
    model.train()
    model.zero_grad()
    logits = model.forward(input,caption_input)
    loss = logits[0]
    loss.backward()
    optimizer.step()

    return loss.item()

def validation_step(input, caption_input, model):
    model.eval()
    logits = model.forward(input, caption_input)
    loss = logits[0]
    
    return loss.item()

def get_validation_loss(batch_size, n_examples, validation_loader, model, device):
    n_iter = int(n_examples/batch_size)
    loader_iter = iter(validation_loader)
    total_loss = 0
    for i in range(n_iter):
        img, cap_in, _, _, img_name = next(loader_iter)
        img = img.to(device)
        cap_in = cap_in.to(device)
        loss = validation_step(img, cap_in, model)
        total_loss += loss
    total_loss = total_loss/n_iter
    return total_loss



if __name__ == "__main__":

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    print('Using {}'.format(device))

    num_workers = os.cpu_count()
    print("Number of CPUs: ", num_workers)


    tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
    train_dataset = RocoDataset(roco_path=roco_path, mode="train", caption_max_length=64, tokenizer=tokenizer)
    valid_dataset = RocoDataset(roco_path=roco_path, mode="validation", caption_max_length=64, tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, drop_last=False, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle_valid, drop_last=False, num_workers=num_workers)
    total_steps = int(len(train_loader)*train_dataset_percentage)  
    print(f"Número total de passos para percorrer {train_dataset_percentage*100}% do dataset: {total_steps}")

    model = ConvNext2T5Model(tokenizer=tokenizer, pretrained_encoder=encoder_name, pretrained_decoder=decoder_name)
    model.n_params()

    train_evaluator = MetricsEvaluator(roco_path=roco_path, mode="train")
    valid_evaluator = MetricsEvaluator(roco_path=roco_path, mode="validation")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if load:
        try:
            torch_load = torch.load(load_path)
            step = torch_load['step']
            best_loss = torch_load['best_loss']
            model.load_state_dict(torch_load)
            del torch_load
            torch.cuda.empty_cache()
            print("Modelo carregado com sucesso!")
        except:
            step = 0
            best_loss = float("inf")
            print("Erro ao carregar os dados. O modelo será inicializado com os dados pré-treinados.")
    else:
        step = 0
        best_loss = float("inf")

    if save_checkpoint:
        save_checkpoint = SaveCheckpoint(save_checkpoint_path, best_valid_loss=best_loss)
    if save_best_model:
        save_best_model = SaveBestModel(save_best_model_path, best_valid_loss=best_loss)

    model = model.to(device)

    if use_neptune_monitoring:

        parameters = {
        "lr": lr,
        "bs": batch_size,
        "model_filename": f"{encoder_name}_{decoder_name}" ,
        "device": device,
        }

        model_name = f"{encoder_name}_{decoder_name}"
        criterion = "loss"
        optimizer_name = "Adam"

        project = neptune_project_name
        neptune_monitoring = NeptuneMonitoring(
            API_TOKEN=NEPTUNE_API_TOKEN,
            project_name=project,
            hyperparameters=parameters,
            model=model_name,
            criterion=criterion,
            optimizer=optimizer_name)

        neptune_monitoring.start()
    

    iter_loader = iter(train_loader)
    valid_loss = 0
    for i in tqdm(range(len(train_loader))):
        try:
            img, caption_input, caption_target, _, _ = next(iter_loader)
            if i < step:
                continue
            if i > total_steps:
                if save_checkpoint:
                    save_checkpoint(valid_loss, step, model, optimizer, "Loss")
                break

            img = img.to(device)
            caption_input = caption_input.to(device)

            loss = train_step(img, caption_input, model)
            perplexity = np.exp(loss)
            if use_neptune_monitoring:
                neptune_monitoring.log_metrics(mode="train", loss=loss, step=step, valid_mode="common")
                neptune_monitoring.log_metrics(mode="train", perplexity=perplexity, step=step, valid_mode="common")
            print(f" Step: {step}, Train Loss: {loss}, Train Perplexity: {perplexity}")


            if step % eval_every_steps == 0:
                print(f"Iniciando validação comum ({eval_every_steps} passos)...")
                model.eval()
                try:
                    
                    valid_loss = get_validation_loss(batch_size, common_validation_batch, valid_loader, model, device)
                    valid_perplexity = np.exp(valid_loss)
                    print(f"Valid loss: {valid_loss}, Valid Perplexity: {valid_perplexity}")
                    if save_best_model:
                        save_best_model(valid_loss, step, model, optimizer, "Loss")
                    if save_checkpoint:
                        print("Salvando Checkpoint...")
                        save_checkpoint(valid_loss, step, model, optimizer, "Loss")

                    if use_neptune_monitoring:
                        neptune_monitoring.log_metrics(mode="validation", loss=valid_loss, step=step, valid_mode="validation")
                        neptune_monitoring.log_metrics(mode="validation", perplexity=valid_perplexity, step=step, valid_mode="validation")
                except Exception as e:
                    print(f"Erro na validação -> {e}")

            if step % refined_eval_every_steps == 0:
                model.eval()
                try:
                    print(f"Iniciando validação refinanada ({refined_eval_every_steps} passos)...")
                    validation_dict = generate_evaluation_dict(model, valid_loader,tokenizer=tokenizer, n_eval=refined_validation_batch, device=device)
                    meteor_score = valid_evaluator.evaluate_meteor(validation_dict)
                    bleu_score = valid_evaluator.evaluate_bleu(validation_dict, case_sensitive=True, stemming=True, remove_stopwords=True)["obtained_score"]
                    rouge_score = valid_evaluator.evaluate_rouge(validation_dict)
                    if use_neptune_monitoring:
                        neptune_monitoring.log_metrics(mode="validation", bleu=bleu_score, step=step, valid_mode="refined")
                        neptune_monitoring.log_metrics(mode="validation", rouge=rouge_score, step=step, valid_mode="refined")
                        neptune_monitoring.log_metrics(mode="validation", meteor=meteor_score, step=step, valid_mode="refined")
                    print(f"Meteor: {meteor_score}")
                    print(f"Bleu: {bleu_score}")
                    print(f"Rouge: {rouge_score}")
                except Exception as e:
                    print(f"Erro na validação refinada -> {e}")
        except Exception as e:
            print(f"Problema no passo {step}. Pulando para o próximo. Erro -> {e}")

        step += 1


    neptune_monitoring.stop()


    ...



