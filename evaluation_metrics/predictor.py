import torch
from tqdm import tqdm
import pandas as pd

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


class PhraseEvaluator():
    def __init__(self, model, tokenizer, max_output_tokens, context_size):
        """
        Instancia classe que facilita a predicao de legendas dada uma imagem
        Inputs:
            model -> Classe que herdou nn.Model a ser avaliada
            tokenizer -> Tokenizador 
            max_output_tokens -> Maior número possível de palavras a serem geradas numa legenda
            context_size -> Tamaho do contexto utilizado no modelo
        """
        self.model = model
        self.tz = tokenizer
        self.max_output_tokens = max_output_tokens
        self.context_size = context_size
    
    def predict_phrase(self, img):
        """
        Prediz uma legenda dada uma imagem
        Inputs:
            img -> Tensor com shape (B, C, H, W) com dimensoes aceitas pelo modelo que foi dado como parametro na inicializacao.
        Outputs:
            prompt (str) -> Legenda predita.
        """
        
        prompt = ''

        for _ in range(self.max_output_tokens):
            input_ids = self.tz.encode(prompt)
            if _ == 0:
                input_ids += (self.context_size - len(input_ids)) * [0]
            input_ids_truncated = input_ids[-self.context_size:]  # Usamos apenas os últimos <context_size> tokens como entrada para o modelo.
            logits = self.model(img.unsqueeze(0),torch.LongTensor([input_ids_truncated]).to(device))
            # Ao usarmos o argmax, a saída do modelo em cada passo é o token de maior probabilidade.
            # Isso se chama decodificação gulosa (greedy decoding).
            predicted_id = torch.argmax(logits).item()
            input_ids += [predicted_id]  # Concatenamos a entrada com o token escolhido nesse passo.
            prompt = self.tz.decode(input_ids, skip_special_tokens=False)

        prompt = self.tz.encode(prompt)
        prompt = self.tz.decode(prompt, skip_special_tokens=True)

        return prompt


def predict_dataset(dataset, evaluator, dataset_range):
    """
    Dado um dataset e um tamanho, faz-se uma predicao de todas as legendas dentro de um range especificado de imagens
    Inputs:
        dataset
        evaluator: classe PhraseEvaluator com todos os parametros desejados declarados previamente
        dataset_range (int): Valor indicando o numero de imagens do dataset que se deseja avaliar
    Returns:
        results_df (DataFrame): retorna um dataframe que organiza o nome da imagem com a legenda gerada, a fim de facilitar o calculo de métricas como BLEU e ROUGE
    """
    results_dict = {}
    for i in tqdm(range(dataset_range)):
        img, cap_in, cap_target, keywords, img_name = dataset[i]
        img = img.to(device)
        prompt = [evaluator.predict_phrase(img)]
        results_dict[img_name] = prompt

    results_df = pd.DataFrame.from_dict(results_dict, orient="index").reset_index()
    results_df.rename(columns={"index":0, 0:1}, inplace=True)

    return results_df
