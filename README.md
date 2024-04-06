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
  - Gensim
- Classification models :
  - NB
  - SVC
  - SGD
  - RF
DNN Models :
- Bert
- French_Bert

A

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
```
python src/run.py --input /path/to/input.csv --output /path/to/output/dir --model SVC --vectorizer TFIDF
```
