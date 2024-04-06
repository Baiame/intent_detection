from joblib import load
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification


from evaluation import compute_metrics


class Model:
    def __init__(self, model: str, weights_path: str, output_dir: str = "."):
        """
        Initialize the model.

        Parameters:
            model: Type of model to use. Allowed values are "NB", "SVC", "RF", "SGD", "SGD_GENSIM", "Bert" or "French_Bert.
            weights_path: path to the saved weights of the model.

        Raises:
            ValueError: Unknown model type is provided.
        """
        self.model_type = model
        if model in ["NB", "SVC", "RF", "SGD", "SGD_GENSIM"]:
            self.model = load(weights_path)
        elif model in ["Bert", "French_Bert"]:
            test_args = TrainingArguments(
                output_dir=output_dir,
                do_train=False,
                do_predict=True,
                per_device_eval_batch_size=16,
                dataloader_drop_last=False,
            )
            model = AutoModelForSequenceClassification.from_pretrained(weights_path)
            self.model = Trainer(
                model=model, args=test_args, compute_metrics=compute_metrics
            )
        else:
            raise ValueError("Unknown model.")

    def predict(self, data):
        """
        Predict labels for the given data.

        Parameters:
            data: The input data for prediction.

        Returns:
            list: The predicted labels.
        """
        return self.model.predict(data)
