
import neptune.new as neptune


class NeptuneMonitoring():
    def __init__(self, hyperparameters:dict, API_TOKEN:str,  project_name:str, model, criterion, optimizer):
        """
        Classe responsável por gerenciar o monitoramento da execução no Neptune 
        Rodar as dependencias abaixo para que o codigo nao tenha prolemas:
            pip install neptune-client numpy==1.19.5 torch==1.9.0 torchvision==0.10.0
        Inputs:
            hyperparameters (dict): dicionário contento hiperparametros relevantes, como leraning rate e batch size, ou qualquer outra coisa desejada
            API_TOKEN (str): Token de acesso ao Neptune gerado ao criar a conta
            project_name (str): Nome do projeto do Neptune em que se armazenará os daddos
            model (str ou torn.nn.Model): Nome do modelo ou a propria classe
            criterion (str ou torn.nn.Criterion): Nome do critério ou a propria classe
            optimizer (str ou torn.nn.Model): Nome do otimizador ou a propria classe
        """
        self.hyperparameters = hyperparameters
        self.API_TOKEN = API_TOKEN
        self.project_name = project_name
        if type(model) == str:
            self.model_name = model
        else:
            self.model_name = type(model).__name__
        if type(criterion) == str:
            self.criterion_name = criterion
        else:
            self.criterion_name = type(criterion).__name__
        if type(optimizer) == str:
            self.optimizer_name = optimizer
        else:
            self.optimizer_name = type(optimizer).__name__

    def start(self):
        """
        Inicia o monitoramento nos servidores do Neptune.
        Assim que inicializado, o tempo de monitoramento estará correndo, portanto é necessário desligar sempre que nao estiver utilizado.
        """
        self.run = neptune.init(
            project=self.project_name, api_token=self.API_TOKEN
        )
        self.run["config/hyperparameters"] = self.hyperparameters
        self.run["config/model"] = self.model_name
        self.run["config/criterion"] = self.criterion_name
        self.run["config/optimizer"] = self.optimizer_name

    def log_metrics(self, step:int, mode=None, bleu=None, rouge=None, perplexity=None, loss=None):
        """
        Grava os logs de métricas 
        Inputs:
            step: Numero de step atual
            mode (train, test, validation): Indica se os dados devem ser armazenados como de teste, validação ou treino
            bleu: Caso diferente de None, salva o score BLEU dessa rodada
            rouge: Caso diferente de None, salva o score ROUGE dessa rodada
            perplexity: Caso diferente de None, salva o score de perplexidade dessa rodada
            loss: Caso diferente de None, salva o score de loss dessa rodada
        """
        if (mode!="train") and (mode!="validation") and (mode!="test"):
            raise("Please, provide a valid mode name (train, validation or test)")
        
        if bleu is not None:
            self.run["training/batch/bleu"].log(bleu)
        if rouge is not None:
            self.run["training/batch/rouge"].log(rouge)
        if perplexity is not None:
            self.run["training/batch/perplexity"].log(perplexity)
        if loss is not None:
            self.run["training/batch/loss"].log(loss)
    
    def stop(self):
        print("Stopping Neptune monitoring...")
        self.run.stop()


if __name__ == "__main__":
    parameters = {
    "lr":3e-5,
    "bs": 5,
    "model_filename": "basemodel",
    "device": "cpu",
    }
    NEPTUNE_API_TOKEN = "teste"

    model = "test_model"
    criterion = "Cross Entropy Loss"
    optimizer = "Adam"

    project="p175857/IA025-Neptune-Test"
    neptune_monitoring = NeptuneMonitoring(
                                        API_TOKEN=NEPTUNE_API_TOKEN,
                                        project_name=project,
                                        hyperparameters=parameters,
                                        model=model,
                                        criterion=criterion,
                                        optimizer = optimizer  )
    neptune_monitoring.start()
    neptune_monitoring.stop()
    