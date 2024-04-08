""" Pool of functions to pre-process and post-process text data."""
import re
import unicodedata

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def tokenize(text: str) -> list[str]:
    """
    Tokenize text into words.

    Args:
        text: text string.

    Returns:
        list of token words.
    """
    return word_tokenize(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """
    Remove stopwords from the tokenized text.

    Args:
        tokens: list of token words.

    Returns:
        List of token words.
    """
    stop_words = set(stopwords.words("french"))
    return [token for token in tokens if token not in stop_words]


def lemmatize(tokens: list[str]) -> list[str]:
    """
    Lemmatize a list of tokens.

    Definition: Lemmatization (or less commonly lemmatisation) in linguistics is
        the process of grouping together the inflected forms of a word so they
        can be analysed as a single item, identified by the word's lemma, or
        dictionary form.

    Args:
        tokens: list of token words.

    Returns:
        List of token words.
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]


def preprocess(text: str) -> list[str]:
    """
    Preprocesses the input text with the following steps :
    1. Converts the text to lowercase.
    2. Normalizes the text using Normalization Form Decomposition.
    3. Removes any characters that are not in the ASCII range.
    4. Removes any non-alphabetic characters except whitespaces.
    5. Tokenizes the text.
    6. Removes stopwords from the tokens.
    7. Lemmatizes the tokens.

    Parameters:
        text: The input text to preprocess.

    Returns:
        The preprocessed text.
    """
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return " ".join(tokens)
