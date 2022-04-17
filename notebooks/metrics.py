from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, ConfusionMatrixDisplay

def display_metrics(y_val, y_pred):
    print('accuracy:', accuracy_score(y_val, y_pred))
    print('precision:', precision_score(y_val, y_pred, average='weighted', zero_division=0))
    print('recall:', recall_score(y_val, y_pred, average='weighted', zero_division=0))
    print('classification repoort:\n', classification_report(y_val, y_pred))

def display_conf_matrix(y_val, y_pred):
    ConfusionMatrixDisplay.from_predictions(y_val, y_pred)
    plt.show()