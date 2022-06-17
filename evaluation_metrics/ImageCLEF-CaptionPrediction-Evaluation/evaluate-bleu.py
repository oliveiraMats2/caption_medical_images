from cgi import test
import sys, argparse, string
import csv, io
from matplotlib.pyplot import text
import nltk
import warnings
import pandas as pd

from nltk.translate.bleu_score import SmoothingFunction
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer



def bleu_evaluator(file_1, file_2, remove_stopwords=False, stemming=False, case_sensitive=False):

    # Hide warnings
    warnings.filterwarnings('ignore')

    # Stats on the captions
    min_words = sys.maxsize
    max_words = 0
    max_sent  = 0
    total_words = 0
    words_distrib = {}

    # NLTK
    # Download Punkt tokenizer (for word_tokenize method)
    # Download stopwords (for stopword removal)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

    # English Stopwords
    stops = set(stopwords.words("english"))

    # Stemming
    stemmer = SnowballStemmer("english")

    # Remove punctuation from string
    translator = str.maketrans('', '', string.punctuation)

    # Parse arguments
    #parser = argparse.ArgumentParser()
    #parser.add_argument('candidate_file', help='path to the candidate file to evaluate')
    #parser.add_argument('gt_file', help='path to the ground truth file')
    #parser.add_argument('-r', '--remove-stopwords', default=False, action='store_true', help='enable stopword removal')
    # parser.add_argument('-s', '--stemming', default=False, action='store_true', help='enable stemming')
    # parser.add_argument('-c', '--case-sensitive', default=False, action='store_true', help='case-sensitive evaluation')
    # args = parser.parse_args()

    # Read files
    # print('Input parameters\n********************************')

    #print('Candidate file is "' + args.candidate_file + '"')
    #candidate_pairs = readfile(args.candidate_file)
    candidate_pairs = readfile(file_1)

    #print('Ground Truth file is "' + args.gt_file + '"')
    #gt_pairs = readfile(args.gt_file)
    gt_pairs = readfile(file_2)

    # print('Removing stopwords is "' + str(args.remove_stopwords) + '"')
    # print('Stemming is "' + str(args.stemming) + '"')

    # Define max score and current score
    max_score = len(gt_pairs)
    current_score = 0

    # Evaluate each candidate caption against the ground truth
    # print('Processing captions...\n********************************')

    i = 0
    for image_key in candidate_pairs:

        # Get candidate and GT caption
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
                bleu_score = nltk.translate.bleu_score.sentence_bleu([gt_words], candidate_words, smoothing_function=SmoothingFunction().method0)
        # Handle problematic cases where BLEU score calculation is impossible
        except ZeroDivisionError:
            print("Problem with zero division")
            #print('Problem with ', gt_words, candidate_words)

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

        # Progress display
        # i += 1
        # if i % 1000 == 0:
        #     print(i, '/', len(gt_pairs), ' captions processed...')

    # Print stats
    # print('Corpus statistics\n********************************')
    # print('Number of words distribution')
    # print_dict_sorted_num(words_distrib)
    # print('Least words in caption :', min_words)
    # print('Most words in caption :', max_words)
    # print('Average words in caption :', total_words / len(gt_pairs))
    # print('Most sentences in caption :', max_sent)

    # Print evaluation result
    # print('Final result\n********************************')
    # print('Obtained score :', current_score, '/', max_score)
    # print('Mean score over all captions :', current_score / max_score)

    results_dict = {
        "min_words": min_words,
        "max_words": max_words,
        "words_distrib": words_distrib,
        "avg_words_in_caption": total_words/len(gt_pairs),
        "most_sentences_in_caption": max_sent,
        "obtained_score": current_score,
        "max_score": max_score,
        "mean_bleu_score": current_score/max_score
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

class BleuEvaluator():
    def __init__(self, roco_path=False, mode="train"):
        """
        Classe utilizada para validar o score BLEU de acordo com as diretrizes da competicao ImageCLEFmedical Caption. 
        Código de avaliacao foi retirado do site da propria competicao e essa classe foi utilizada para facilitar a integracao dos testes com o treino da rede neural.
        Inputs:
            roco_path: str -> diretorio em que o dataset roco está armazenado
            mode: str -> (train, validation, test) - seleciona qual particao do dataset sera utilizada na avaliacao
        """
        text_buffer = io.StringIO()
        if roco_path != False:
            file_path = roco_path + f"/data/{mode}/radiology/captions.txt"
            df_reference = pd.read_csv(file_path, sep="\t", header=None)
            df_reference.to_csv(text_buffer, sep="\t", index=False)
        self.reference_bin = text_buffer.getvalue()
    
    def __call__(self, df_candidate: pd.DataFrame, remove_stopwords:bool=False, stemming:bool=False, case_sensitive:bool=False):
        """
        Calcula o score BLEU e retorna todas as informacoes obtidas num dicionario.
        Inputs: 
            df_candidate: DataFrame -> Tabela contendo as predicoes a serem avaliadas. Vale notar que ela precisa estar no mesmo formato tabela captions.txt, contendo as mesmas keys referenciando os nomes das imagens.
            remove_stopwords: bool -> Define se haverá remocao de stopwords na avaliacao.
            stemming: bool -> Define se haverá stemming na avaliacao.
            case_sensitive: bool -> Define se a avaliacao diferenciará letras maiúsculas de minúsculas.
        """
        text_buffer = io.StringIO()
        df_candidate.to_csv(text_buffer, sep="\t", index=False)
        candidate_bin = text_buffer.getvalue()
        evaluation_dict = bleu_evaluator(self.reference_bin, candidate_bin, remove_stopwords=remove_stopwords, stemming=stemming, case_sensitive=case_sensitive)
        return evaluation_dict



if __name__ == '__main__':
    # Exemplo de utilizacao da classe BleuEvaluator.
    # Neste caso o df_evaluation é o mesmo que o arquivo de referencia apenas para fins de teste. Em um caso real, este dataframe deveria ser construído a partir da predicao da rede neural.
    path_file = "/Users/pdcos/Documents/Mestrado/IA025/Projeto_Final/Code/caption_medical_images/dataset_structure/roco-dataset/data/test/radiology/captions.txt"
    df_evaluation = pd.read_csv(path_file, sep="\t", header=None)
    text_buffer = io.StringIO()
    df_evaluation.to_csv(text_buffer, sep="\t", index=False)
    text_buffer_content = text_buffer.getvalue()

    roco_path = "/Users/pdcos/Documents/Mestrado/IA025/Projeto_Final/Code/caption_medical_images/dataset_structure/roco-dataset/"
    bleu = BleuEvaluator(roco_path=roco_path, mode="test")
    results = bleu(df_evaluation, case_sensitive=True, stemming=True, remove_stopwords=True)
    print(results)
