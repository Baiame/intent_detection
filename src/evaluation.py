from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from schemas import LABEL_DEF
from pathlib import Path


def compute_metrics(pred) -> dict:
    """
    Compute accuracy metric based on predictions.

    Parameters:
        pred: The predictions.

    Returns:
        A dictionary containing the accuracy metric.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


def perform_eval(preds: list[str], labels: list[str], output_dir: str):
    """
    Perform evaluation and save evaluation results.

    Parameters:
        preds: Predicted labels.
        labels: True labels.
        output_dir: Directory to save evaluation results.
    """
    report = classification_report(labels, preds)
    with open(Path(output_dir, "report.txt"), "w") as file:
        file.write(report)
    save_confusion_matrix(preds, labels, output_dir)
    print(report)


def save_confusion_matrix(preds: list[str], labels: list[str], output_dir: str):
    """
    Save confusion matrix plot.

    Parameters:
        preds: Predicted labels.
        labels: True labels.
        output_dir: Directory to save the confusion matrix plot.
    """
    available_labels = list(LABEL_DEF.keys())
    conf_matrix = confusion_matrix(preds, labels, labels=available_labels)
    _, _ = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=available_labels,
        yticklabels=available_labels,
    )
    plt.xlabel("Predicted Labels")
    plt.ylabel("Actual Labels")
    plt.title("Confusion Matrix")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(output_dir, "cm.png"))
