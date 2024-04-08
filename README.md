# French Intent Detection

Perform intent detection on given any given French sentence or text dataset.

The classification will be done in 8 classes :
- `translate`: the user wants to translate a sentence into another language
- `travel_alert`: the user asks if their destination is affected by a travel alert
- `flight_status`: the user asks for information about the status of their flight
- `lost_luggage`: the user reports the loss of their luggage
- `travel_suggestion`: the user wants a travel recommendation
- `carry_on`: the user wants information about carry-on luggage
- `book_hotel`: the user wants to book a hotel
- `book_flight`: the user wants to book a flight

## Install

### Clone

```bash
git clone git@github.com:Baiame/intent_detection.git
cd intent_detection
```

### Python

Have Python `>=3.9` installed.
Have `make` installed. For example using `brew`:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install make
```

Install requirements:
```bash
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

or just :
```
make install
```

## Models
Available models are :
- Vectorizers :
  - [TFIDF](https://medium.com/@cmukesh8688/tf-idf-vectorizer-scikit-learn-dbc0244a911a)
  - [Gensim](https://fauconnier.github.io/)
- Classification Models :
  - [NB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
  - [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
  - [SGD](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
  - [RF](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- DNN Models (no vectorizer needed) :
  - [Bert](https://huggingface.co/albert/albert-base-v2)
  - [French_Bert](https://huggingface.co/dbmdz/bert-base-french-europeana-cased)

All the models are available in the repository, except for 2 models :

- For the French_Bert models, fine-tuned weights are available [at his link](https://we.tl/t-FxKdwjOnw4).
Just download the weights and place the folder in the models repository, following this structure :

```md
intent_detection/
├── models/
│   ├── svd.joblib
│   ├── ...
│   ├── albert-base-v2/
│   │   ├── config.json
│   │   └── model.safetensor
│   └── french-bert/
│       ├── config.json
│       └── model.safetensor
├── src/
...
```

- For the Gensim vectorizer weights, run this command :
```
wget https://embeddings.net/embeddings/frWiki_no_phrase_no_postag_700_cbow_cut100.bin
```

## Run
### Run interactive shell
```
python src/run.py --interactive True
```

### Inference on an input sentence
```
python src/run.py --text [SENTENCE] --model [MODEL] --vectorizer [VECTORIZER]
```

For example :
```
python src/run.py --text "A quelle heure doit décoller le vol AF345 ?" --model French_Bert
```

### Evaluation
The input CSV file must have 2 columns: `text` and `label`.

For example :
|text|label|
|---|---|
|Quelle est l'heure de décollage du prochain vol pour Paris ?|flight_status|
|J'ai besoin de réserver une chambre à Lisbonne|book_hotel|

To run the evaluation, you can use :
```
python src/run.py --input /path/to/input.csv --output /path/to/output/dir --model SVC --vectorizer TFIDF
```
After evaluation, the output directory will contain the predictons, the metrics and a confusion matrix.