# Plot the Graph
# Loss Curves
import matplotlib.pyplot as plt
import pandas as pd

path_init = 'results_csv/trained_cluster'
path_retrain_init = 'results_csv/re_trained_cluster'

train_loss_retrain = pd.read_csv(f'{path_retrain_init}/validation_loss_score.csv')
perplexity_score_retrain = pd.read_csv(f'{path_retrain_init}/validation_perplexity_score.csv')

valid_loss_init = pd.read_csv(f'{path_init}/validation_loss_score.csv')
perplexity_score_init = pd.read_csv(f'{path_init}/validation_perplexity_score.csv')
#
plt.figure(figsize=[20, 8])
plt.plot(train_loss_retrain['position'], train_loss_retrain['value'], 'b', linewidth=3.0)
plt.plot(valid_loss_init['position'], valid_loss_init['value'], 'r', linewidth=3.0)
plt.legend(['validação Retreinamento [loss]', 'validação inicio do treinamento [loss]'], fontsize=18)
plt.xlabel('Epocas', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Curvas função de perda', fontsize=16)
#
# # Accuracy Curves
plt.figure(figsize=[20, 8])
plt.plot(perplexity_score_retrain['position'], perplexity_score_retrain['value'], 'b', linewidth=3.0)
plt.plot(perplexity_score_init['position'], perplexity_score_init['value'], 'r', linewidth=3.0)
plt.legend(['validação Retreinamento [perplexity]', 'validação inicio do treinamento [perplexity]'], fontsize=18)
plt.xlabel('Epocas', fontsize=16)
plt.ylabel('perplexidade', fontsize=16)
plt.title('Curvas de perplexidade', fontsize=16)
plt.show()