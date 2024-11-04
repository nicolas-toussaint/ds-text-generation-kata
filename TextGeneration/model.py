from collections import Counter
import random

import joblib
from TextGeneration.utils.preprocessor import Preprocessor


class NGramsModel:
    """A model used to generate text from text input.


    :param max_n_gram: The max n gram

    """

    def __init__(self, max_n_gram: int) -> None:
        self.max_n_gram = max_n_gram

        # Choose a random prediction from the n top candidates. Default to 1.
        self.use_top_candidate = 1

        # Only use ngram when they have been found n times in the training dataset.
        self._ngram_minimal_count = 6

        self._data = {}

    def _ngrams(self, text: str, ngram_size: int) -> list[tuple[str]]:
        """Create ngrams from a text.

        :param text: Text to create ngrams from
        :param ngram_size: The size of the ngrams to create
        :return: Ngrams created from the given text
        """

        words = text.split()
        return [
            tuple(words[i : i + ngram_size]) for i in range(len(words) - ngram_size + 1)
        ]

    def _add_ngrams(self, ngrams: list[tuple[str]]) -> None:
        """Add the ngrams to the model data.

        The model data will be used to generate text from the
        saved ngrams.

        :param ngrams: The ngrams to add
        :return: None
        """

        for ngram in ngrams:
            ngram_key = ngram[:-1]
            ngram_candidate_name = ngram[-1]

            ngram_candidates = self._data.setdefault(ngram_key, Counter())
            ngram_candidates.update({ngram_candidate_name: 1})

    def partial_fit(self, text: str) -> None:
        """Online learnig from a stream of examples.

        :param ngrams: The ngrams to add
        :return: None
        """

        for i in range(2, self.max_n_gram + 1):
            ngrams = self._ngrams(text, i)
            self._add_ngrams(ngrams)

    def predict(self, text_input: str) -> str:
        """Generate text from a text input.

        :param text_input: The input used to generate text
        :return: The generated text
        """
        result = text_input.split()[:-1]

        for _ in range(50):
            query = tuple(result[-(self.max_n_gram - 1) :])

            for i in range(len(query), 0, -1):
                if candidates := self._data.get(query[-i:]):

                    filtred_candidates = [
                        name
                        for name, count in candidates.most_common(
                            self.use_top_candidate
                        )
                        if count >= self._ngram_minimal_count
                    ]

                    if filtred_candidates:
                        next_word = random.choice(filtred_candidates)

                        result.append(next_word)
                        break
            else:
                break

        return self._clean_prediction(result)

    def _clean_prediction(self, words: list[str]) -> str:
        """Clean the generate prediction.

        :param words: The prediction
        :return: The cleaned prediction
        """

        result = " ".join(words)
        result = result.removeprefix(Preprocessor.text_prefix)
        result = result.removesuffix(Preprocessor.text_suffix)

        return result

    def save(self, trained_model: str) -> None:
        """Save the model to a file.

        :param trained_model: The file name
        :return: None
        """

        joblib.dump(self._data, trained_model)

    @classmethod
    def load(cls, trained_model: str, max_n_gram: int) -> "NGramsModel":
        """Load a model from a file.

        :param trained_model: The file name
        :param max_n_gram: The max n gram
        :raised: FileNotFoundError if the given path has not been found.
        :return: A model
        """

        model = NGramsModel(max_n_gram)
        try:
            data = joblib.load(trained_model)
        except FileNotFoundError as err:
            raise FileNotFoundError(
                f"The model dump file '{trained_model}' has not be found. Please check that "
                "the model has been trained beforehand and that the given model path is the same."
            ) from err

        model._data = data

        return model
