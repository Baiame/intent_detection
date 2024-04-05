from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Word2VecKeyedVectors
import pickle
import numpy as np


class Vectorizer:
    """ Vectorizer class. """
    def __init__(self, vectorizer, weights_path) -> None:
        """ Instantiate the vectorizer model from the weights path.
        
        Args:
            vectorizer: the vectorizer name.
            weights_path: path to the weights file.
        """

        self.vectorizer_type = vectorizer

        if vectorizer == "TFIDF":
            self.vectorizer = pickle.load(open(weights_path, "rb"))
        elif vectorizer == "GENSIM":
            self.vectorizer = KeyedVectors.load_word2vec_format(
                weights_path, binary=True, unicode_errors="ignore"
            )
        else:
            raise ValueError("Unknown vectorizer.")

    def transform(self, data: list[str]):
        """
        Apply vectorization to the input data.

        Parameters:
            data: The list of sentences to transform.

        Returns:
            Vectorized representation of the input data.
        """
        if self.vectorizer_type == "TFIDF":
            return self.vectorizer.transform(data)
        if self.vectorizer_type == "GENSIM":
            embedding_dim = self.vectorizer.vector_size
            return _vectorize(data, self.vectorizer, embedding_dim)


def _vectorize_sentence(sentence: str, model: Word2VecKeyedVectors, embedding_dim: int) -> np.ndarray:
    """
    Vectorizes a sentence using the provided Word2Vec model.

    Parameters:
        sentence: The input sentence to vectorize.
        model: The Word2Vec model used for vectorization.
        embedding_dim: The dimensionality of the word embeddings.

    Returns:
        The vectorized representation of the sentence.
    """
    vectorized_sentence = []
    for word in sentence.split():
        if word in model.vocab:
            vectorized_sentence.append(model[word])
    if len(vectorized_sentence) == 0: #all words are not found in the model
        return np.zeros(
            embedding_dim
        ) 
    else:
        return np.mean(vectorized_sentence, axis=0)

def _vectorize(data: list[str], model: Word2VecKeyedVectors, embedding_dim: int) -> np.ndarray:
    """
    Vectorizes a list of sentences using the provided Word2Vec model.

    Parameters:
        data: The list of sentences to vectorize.
        model: The Word2Vec model used for vectorization.
        embedding_dim: The dimensionality of the word embeddings.

    Returns:
        The matrix of vectorized representations of the sentences.
    """
    vectors = []
    for sentence in data:
        vectorized_sentence = _vectorize_sentence(sentence, model, embedding_dim)
        vectors.append(vectorized_sentence)
    return np.array(vectors)
