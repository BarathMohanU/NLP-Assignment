import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import seaborn as sns

data = loadmat('./Saved Metrics/lstm_model_history.mat')
svm_data = loadmat('./Saved Metrics/svm_metrics.mat')

x = np.arange(100) + 1

plt.plot(x, np.squeeze(data['loss']), color=sns.color_palette()[0], label='LSTM Train Loss')
plt.plot(x, np.squeeze(data['val_loss']), color=sns.color_palette()[1], label='LSTM Test Loss')
plt.ylabel('Cross Entropy Loss')
plt.xlabel('Epochs')
plt.legend()
plt.savefig('./Plots/loss_plot.png', dpi=300)
plt.show()

plt.plot(x, np.squeeze(data['precision']), color=sns.color_palette()[0], label='LSTM Train Precision')
plt.plot(x, np.squeeze(data['val_precision']), color=sns.color_palette()[1], label='LSTM Test Precision')
plt.axhline(y = svm_data['train_precision'], color = sns.color_palette()[0], linestyle = '--', label='SVM Train Precision')
plt.axhline(y = svm_data['test_precision'], color = sns.color_palette()[1], linestyle = '--', label='SVM Test Precision')
plt.ylabel('Precision')
plt.xlabel('Epochs')
plt.legend()
plt.savefig('./Plots/precision_plot.png', dpi=300)
plt.show()

plt.plot(x, np.squeeze(data['recall']), color=sns.color_palette()[0], label='LSTM Train Recall')
plt.plot(x, np.squeeze(data['val_recall']), color=sns.color_palette()[1], label='LSTM Test Recall')
plt.axhline(y = svm_data['train_recall'], color = sns.color_palette()[0], linestyle = '--', label='SVM Train Recall')
plt.axhline(y = svm_data['test_recall'], color = sns.color_palette()[1], linestyle = '--', label='SVM Test Recall')
plt.ylabel('Recall')
plt.xlabel('Epochs')
plt.legend()
plt.savefig('./Plots/recall_plot.png', dpi=300)
plt.show()

plt.plot(x, np.squeeze(data['accuracy']), color=sns.color_palette()[0], label='LSTM Train Accuracy')
plt.plot(x, np.squeeze(data['val_accuracy']), color=sns.color_palette()[1], label='LSTM Test Accuracy')
plt.axhline(y = svm_data['train_accuracy'], color = sns.color_palette()[0], linestyle = '--', label='SVM Train Accuracy')
plt.axhline(y = svm_data['test_accuracy'], color = sns.color_palette()[1], linestyle = '--', label='SVM Test Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.savefig('./Plots/accuracy_plot.png', dpi=300)
plt.show()

plt.plot(x, np.squeeze(data['auc']), color=sns.color_palette()[0], label='LSTM Train AUROC')
plt.plot(x, np.squeeze(data['val_auc']), color=sns.color_palette()[1], label='LSTM Test AUROC')
plt.axhline(y = svm_data['train_auroc'], color = sns.color_palette()[0], linestyle = '--', label='SVM Train AUROC')
plt.axhline(y = svm_data['test_auroc'], color = sns.color_palette()[1], linestyle = '--', label='SVM Test AUROC')
plt.ylabel('AUROC')
plt.xlabel('Epochs')
plt.legend()
plt.savefig('./Plots/auc_plot.png', dpi=300)
plt.show()