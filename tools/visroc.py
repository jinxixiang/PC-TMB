import sklearn.metrics as metrics
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def visroc():
    # plot auc curve
    fpr, tpr, threshold = metrics.roc_curve(labels, prob[:, 1])
    roc_auc = metrics.auc(fpr, tpr)

    # method I: plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    plt.savefig("huayin_roc.jpg")


if __name__ == "__main__":
    preds = np.load()
    labels = np.load()
    visroc()

