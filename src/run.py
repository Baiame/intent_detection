""" Inference entrypoint."""

import argparse
import pandas as pd
from schemas import EVAL_CSV_SCHEMA, INFERENCE_CSV_SCHEMA, LABEL_DEF
from data_processing import preprocess
from vectorizer import Vectorizer
from model import Model
from evaluation import perform_eval
from pathlib import Path

from dataset import ClassificationDataset
from transformers import AutoTokenizer


WEIGHT_PATHS = {
    "TFIDF": "models/tfidf.pickle",
    "GENSIM": "models/frWiki_no_phrase_no_postag_700_cbow_cut100.bin",
    "SGD": "models/sgd.joblib",
    "SGD_GENSIM": "models/sgd_gensim.joblib",
    "SVC": "models/svc.joblib",
    "RF": "models/random_forest.joblib",
    "NB": "models/multinomial_nb.joblib",
    "Bert": "models/albert-base-v2"
}


def main():
    parser = argparse.ArgumentParser(
        description="CLI for text intent classification inference."
    )
    parser.add_argument("--input", type=str, help="Path to the input csv.")
    parser.add_argument("--output", type=str, help="Path to the output dir.")
    parser.add_argument("--text", type=str, help="Text to infer intent.")
    parser.add_argument(
        "--model",
        type=str,
        help="Model type. Available: ['NB', 'SVC', 'RF', 'SGD', 'Bert']",
    )
    parser.add_argument(
        "--vectorizer",
        type=str,
        help="Vectorizer type. Available: ['TFIDF', 'Gensim']",
    )
    args = parser.parse_args()

    eval_mode = False

    # Load data
    if args.input:
        dataset = pd.read_csv(args.input)
        if "label" in dataset.columns:
            dataset = EVAL_CSV_SCHEMA.validate(dataset)
            eval_mode = True
        else:
            dataset = INFERENCE_CSV_SCHEMA.validate(dataset)
            
    else:
        dataset = pd.DataFrame({"text": [args.text]})

    # Pre-processing
    dataset = dataset.drop_duplicates()
    dataset["text"] = dataset["text"].apply(preprocess)

    # Instantiate Model
    if args.model in ["NB", "SVC", "RF", "SGD", "SGD_GENSIM"]:
        model = Model(args.model, weights_path=WEIGHT_PATHS[args.model])
        vectorizer = Vectorizer(
            args.vectorizer, weights_path=WEIGHT_PATHS[args.vectorizer]
        )
        vectors = vectorizer.transform(list(dataset["text"]))
        result = model.predict(vectors)
    elif args.model in ["Bert"]:
        tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
        model = Model(args.model, weights_path=WEIGHT_PATHS[args.model])

        texts_encoded = tokenizer(list(dataset["text"]), padding=True, truncation=True, return_tensors="pt")
        result = model.predict(ClassificationDataset(texts_encoded, [None]*len(dataset)))
        result = result.predictions.argmax(-1)
        invert_map = {v: k for k, v in LABEL_DEF.items()}
        result = [invert_map[res] for res in result]


    else:
        raise ValueError("Unknown model.")

    # If eval:
    if eval_mode:
        perform_eval(result, dataset["label"], args.output)

    if args.output:
        output_dataframe = pd.DataFrame({"results": result})
        output_dataframe.to_csv(Path(args.output, "result.csv"))
    else:
        print(result)

if __name__ == "__main__":
    main()
