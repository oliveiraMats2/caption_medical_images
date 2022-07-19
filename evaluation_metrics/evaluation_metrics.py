from cgi import test
from os import rename
import sys, argparse, string
import csv, io
from matplotlib.pyplot import text
import nltk
import warnings
import pandas as pd
from regex import F
from tqdm import tqdm
from nltk.translate.bleu_score import SmoothingFunction
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from rouge_score import rouge_scorer

import spacy

import datasets



def bleu_evaluator(file_1, file_2, remove_stopwords=False, stemming=False, case_sensitive=False):
    # Hide warnings
    warnings.filterwarnings('ignore')

    # Stats on the captions
    min_words = sys.maxsize
    max_words = 0
    max_sent = 0
    total_words = 0
    words_distrib = {}

    # English Stopwords
    stops = set(stopwords.words("english"))

    # Stemming
    stemmer = SnowballStemmer("english")

    # Remove punctuation from string
    translator = str.maketrans('', '', string.punctuation)


    candidate_pairs = readfile(file_1)
    gt_pairs = readfile(file_2)
    max_score = len(gt_pairs)
    current_score = 0

    i = 0
    print("Calculando score BLEU...")
    for image_key in tqdm(candidate_pairs, total=len(candidate_pairs)):

        # Get candidate and GT caption
        try:
            candidate_caption = candidate_pairs[image_key]
            gt_caption = gt_pairs[image_key]

            # Optional - Go to lowercase
            if not case_sensitive:
                candidate_caption = candidate_caption.lower()
                gt_caption = gt_caption.lower()

            # Split caption into individual words (remove punctuation)
            candidate_words = nltk.tokenize.word_tokenize(candidate_caption.translate(translator))
            gt_words = nltk.tokenize.word_tokenize(gt_caption.translate(translator))

            # Corpus stats
            total_words += len(gt_words)
            gt_sentences = nltk.tokenize.sent_tokenize(gt_caption)

            # Optional - Remove stopwords
            if remove_stopwords:
                candidate_words = [word for word in candidate_words if word.lower() not in stops]
                gt_words = [word for word in gt_words if word.lower() not in stops]

            # Optional - Apply stemming
            if stemming:
                candidate_words = [stemmer.stem(word) for word in candidate_words]
                gt_words = [stemmer.stem(word) for word in gt_words]

            # Calculate BLEU score for the current caption
            try:
                # If both the GT and candidate are empty, assign a score of 1 for this caption
                if len(gt_words) == 0 and len(candidate_words) == 0:
                    bleu_score = 1
                # Calculate the BLEU score
                else:
                    bleu_score = nltk.translate.bleu_score.sentence_bleu([gt_words], candidate_words,
                                                                        smoothing_function=SmoothingFunction().method0)
            # Handle problematic cases where BLEU score calculation is impossible
            except ZeroDivisionError:
                print("Problem with zero division")
                # print('Problem with ', gt_words, candidate_words)

            # Increase calculated score
            current_score += bleu_score
            nb_words = str(len(gt_words))
            if nb_words not in words_distrib:
                words_distrib[nb_words] = 1
            else:
                words_distrib[nb_words] += 1

            # Corpus stats
            if len(gt_words) > max_words:
                max_words = len(gt_words)

            if len(gt_words) < min_words:
                min_words = len(gt_words)

            if len(gt_sentences) > max_sent:
                max_sent = len(gt_sentences)
        except:
            ...

    results_dict = {
        "min_words": min_words,
        "max_words": max_words,
        "words_distrib": words_distrib,
        "avg_words_in_caption": total_words / len(gt_pairs),
        "most_sentences_in_caption": max_sent,
        "obtained_score": current_score,
        "max_score": max_score,
        "mean_bleu_score": current_score / max_score
    }

    return results_dict


def readfile(file):
    path = "file"
    try:
        pairs = {}
        reader = csv.reader(file.splitlines(), delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            pairs[row[0]] = row[1]

        return pairs
    except FileNotFoundError:
        print('File "' + path + '" not found! Please check the path!')
        exit(1);


# Print 1-level key-value dictionary, sorted (with numeric key)
def print_dict_sorted_num(obj):
    keylist = [int(x) for x in list(obj.keys())]
    keylist.sort()
    for key in keylist:
        print(key, ':', obj[str(key)])


class MetricsEvaluator():
    def __init__(self, roco_path=False, mode="train"):
        """
        Classe utilizada para validar o score BLEU e ROUGE de acordo com as diretrizes da competicao ImageCLEFmedical Caption. 
        Código de avaliacao foi retirado do site da propria competicao e essa classe foi utilizada para facilitar a integracao dos testes com o treino da rede neural.
        Inputs:
            roco_path: str -> diretorio em que o dataset roco está armazenado
            mode: str -> (train, validation, test) - seleciona qual particao do dataset sera utilizada na avaliacao
        """
        text_buffer = io.StringIO()
        if roco_path != False:
            file_path = roco_path + f"/data/{mode}/radiology/captions.txt"
            self.df_reference = pd.read_csv(file_path, sep="\t", header=None)
            self.df_reference.to_csv(text_buffer, sep="\t", index=False)

    
        rename_dict={0:"img_name", 1:"phrase_target"}
        self.df_reference.rename(columns=rename_dict, inplace=True)
        self.reference_bin = text_buffer.getvalue()

        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)

        self.meteor = datasets.load_metric('meteor')


    def evaluate_bleu(self, evaluation_dict:dict, remove_stopwords: bool = False, stemming: bool = False,
                      case_sensitive: bool = False):
        """
        Calcula o score BLEU e retorna todas as informacoes obtidas num dicionario.
        Inputs: 
            evaluation_dict: dict -> Dicionário contendo as predicoes a serem avaliadas. Vale notar que ela precisa estar no mesmo formato tabela captions.txt, contendo as mesmas keys referenciando os nomes das imagens.
            remove_stopwords: bool -> Define se haverá remocao de stopwords na avaliacao.
            stemming: bool -> Define se haverá stemming na avaliacao.
            case_sensitive: bool -> Define se a avaliacao diferenciará letras maiúsculas de minúsculas.
        """

        df_evaluation = pd.DataFrame.from_dict(evaluation_dict, orient='index')
        df_evaluation.reset_index(inplace=True)
        rename_dict={"index":"img_name", 0:"phrase_prediction"}
        df_evaluation.rename(columns=rename_dict, inplace=True)


        df_merge = pd.merge(self.df_reference, df_evaluation, how="inner", on=["img_name"])
        df_reference_filtered = df_merge[["img_name", "phrase_target"]]
        df_evaluation_filtered = df_merge[["img_name", "phrase_prediction"]]


        text_buffer = io.StringIO()
        df_reference_filtered.to_csv(text_buffer, sep="\t", index=False, header=None)
        reference_bin = text_buffer.getvalue()
        text_buffer.flush()
        df_evaluation_filtered.to_csv(text_buffer, sep="\t", index=False, header=None)
        evaluation_bin = text_buffer.getvalue()
        evaluation_dict = bleu_evaluator(reference_bin, evaluation_bin, remove_stopwords=remove_stopwords,
                                         stemming=stemming, case_sensitive=case_sensitive)
        return evaluation_dict

    def evaluate_rouge(self, evaluation_dict: dict):
        """
        Calcula o score ROUGE e retorna o fmeasure de unigramas.

        Convenções da competição ImageCLEFmedical a respeito desta métrica:
            The caption is converted to lower-case
            Stopwords are removed using NLTK's "english" stopword list
            Lemmatization is applied using spacy's Lemmatizer

            Fonte: https://www.imageclef.org/2022/medical/caption

        Para rodar essa este método, é necessário instalar alguns pacotes spacy usados para lematizar as palavras. Seguem abaixo os comandos:
            python -m spacy download en
            python -m spacy download en_core_web_sm

        A execução desta métrica é lenta por conta da lematização.

        Inputs:
            evaluation_dict: dict -> Dicionário contendo as predicoes a serem avaliadas. Vale notar que ela precisa estar no mesmo formato tabela captions.txt, contendo as mesmas keys referenciando os nomes das imagens.
        Returns:
            f1_final_score: float -> Score ROUGE final resultado da média de todos os scores de cada linha da tabela de entrada.
        """

        print("Calculando score ROUGE...")

        # Pré processamento para fazer o merge das tabelas de entrada, de modo termos as frases candidatas e de referencia na mesma linha.
        stops = set(stopwords.words("english"))
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)



        df_candidate = pd.DataFrame.from_dict(evaluation_dict, orient='index')
        df_candidate.reset_index(inplace=True)
        rename_dict={"index":"img_name", 0:"phrase_prediction"}
        df_candidate.rename(columns=rename_dict, inplace=True)


        
        df_merge = pd.merge(self.df_reference, df_candidate, how="inner", on=["img_name"])

        load_model = spacy.load("en_core_web_sm", disable=["parser", "ner"])

        f1_scores_list = []
        for index, row in tqdm(df_merge.iterrows(), total=df_merge.shape[0]):
            try:
                # Converte todas as letras para minúsculo e remove a pontuação.
                target = row["phrase_target"].lower()
                prediction = row["phrase_prediction"].lower()
                translator = str.maketrans('', '', string.punctuation)

                # Remove stopwords
                target_words = nltk.tokenize.word_tokenize(target.translate(translator))
                prediction_words = nltk.tokenize.word_tokenize(prediction.translate(translator))
                target_words = [word for word in target_words if word.lower() not in stops]
                prediction_words = [word for word in prediction_words if word.lower() not in stops]
                target_phrase = " ".join(target_words)
                prediction_phrase = " ".join(prediction_words)

                # Faz a lematização (maior custo computacional desta métrica)
                target_lemma = load_model(target_phrase)
                prediction_lemma = load_model(prediction_phrase)
                target_phrase = " ".join([token.lemma_ for token in target_lemma])
                prediction_phrase = " ".join([token.lemma_ for token in prediction_lemma])

                # Calcula os scores e seleciona fmeasure, que é a métrica utilizada no ImageCLEFmedical
                scores = scorer.score(prediction_phrase, target_phrase)
                f1score = scores["rouge1"].fmeasure

                f1_scores_list.append(f1score)
            except:
                continue

        # Tira a média de todas os scores calculados para cada linha da tabela.
        f1_final_socore = sum(f1_scores_list) / len(f1_scores_list)
        return f1_final_socore

    def evaluate_meteor(self, evaluation_dict: dict):
        """
        Cálculo do meteor recebendo um dicionário com diversas predicoes e nome de imagens como chave
        """
        df_candidate = pd.DataFrame.from_dict(evaluation_dict, orient='index')
        df_candidate.reset_index(inplace=True)
        rename_dict={"index":"img_name", 0:"phrase_prediction"}
        df_candidate.rename(columns=rename_dict, inplace=True)

        df_merge = pd.merge(self.df_reference, df_candidate, how="inner", on=["img_name"])
        self.meteor.add_batch(predictions=df_merge["phrase_prediction"].values, 
                                references=df_merge["phrase_target"].values)
        meteor_score = self.meteor.compute()["meteor"]
        return meteor_score

    def evaluate_meteor_fast(self, evaluation_phrase:str, img_name:str):
        """
        Cálculo de meteor para apenas uma frase com intuito de monitorar todos os passos de treinamento.
        """
        reference_phrase = self.df_reference.loc[self.df_reference["img_name"] == img_name]["phrase_target"].values
        self.meteor.add(reference=reference_phrase, prediction=evaluation_phrase)
        meteor_score = self.meteor.compute()["meteor"]
        return meteor_score


if __name__ == '__main__':
    # Exemplo de utilizacao da classe BleuEvaluator.
    # Neste caso o df_evaluation é o mesmo que o arquivo de referencia apenas para fins de teste. Em um caso real, este dataframe deveria ser construído a partir da predicao da rede neural.

    roco_path = "/Users/pdcos/Documents/Mestrado/IA025/Projeto_Final/Code/caption_medical_images/dataset_structure/roco-dataset"
    evaluation_dict = {'ROCO_03160': 'Chest X-ray showing bilateral pleural effusions.',
                        'ROCO_06092': 'Chest X-ray showing bilateral pleural effusions.',
                        'ROCO_09308': 'Chest X-ray showing bilateral pleural effusions.',
                        'ROCO_13014': 'Chest X-ray showing bilateral pleural effusions.',
                        'ROCO_18575': 'Chest X-ray showing bilateral pleural effusions.',
                        'ROCO_21398': 'Chest X-ray showing bilateral pleural effusions.',
                        'ROCO_23923': 'Chest X-ray showing bilateral pleural effusions.',
                        'ROCO_26482': 'Chest X-ray showing bilateral pleural effusions.',
                        'ROCO_30572': 'Chest X-ray showing bilateral pleural effusions.',
                        'ROCO_31053': 'Chest X-ray showing bilateral pleural effusions.',
                        'ROCO_49800': 'Chest X-ray showing bilateral pleural effusions.',
                        'ROCO_53445': 'Chest X-ray showing bilateral pleural effusions.',
                        'ROCO_56532': 'Chest X-ray showing bilateral pleural effusions.',
                        'ROCO_60736': 'Chest X-ray showing bilateral pleural effusions.',
                        'ROCO_62950': 'Chest X-ray showing bilateral pleural effusions.',
                        'ROCO_72035': 'Chest X-ray showing bilateral pleural effusions.',
                        'ROCO_74981': 'Chest X-ray showing bilateral pleural effusions.',
                        'ROCO_75079': 'Chest X-ray showing bilateral pleural effusions.',
                        'ROCO_75557': 'Chest X-ray showing bilateral pleural effusions.',
                        'ROCO_80302': 'Chest X-ray showing bilateral pleural effusions.'}
                        
    evaluator = MetricsEvaluator(roco_path=roco_path, mode="test")
    bleu = evaluator.evaluate_bleu(evaluation_dict, case_sensitive=True, stemming=True, remove_stopwords=True)
    print(bleu)
    rouge = evaluator.evaluate_rouge(evaluation_dict)
    print(rouge)
    meteor = evaluator.evaluate_meteor(evaluation_dict)
    print(meteor)
    meteor = evaluator.evaluate_meteor_fast('Chest X-ray showing bilateral pleural effusions.', 'ROCO_03160')
    print(meteor)

