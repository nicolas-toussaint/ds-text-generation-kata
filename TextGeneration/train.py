"""Code for calling the training of the model."""

from sys import argv
import logging

from TextGeneration.utils.files import json_to_schema, read_dir
from TextGeneration.utils.preprocessor import Preprocessor
from TextGeneration.utils.schemas import TrainingInputSchema
from TextGeneration.model import NGramsModel


def main_train(file_str_path: str) -> None:
    """
    Call for training an n-gram language model.

    Do not modify its signature.
    You can modify the content.

    :param file_str_path: The path to the JSON that configures the training
    :return: None
    """
    # Reading input data
    training_schema = json_to_schema(
        file_str_path=file_str_path, input_schema=TrainingInputSchema
    )

    # Initialize a model
    model = NGramsModel(
        max_n_gram=training_schema.max_n_gram,
    )

    for training_line in read_dir(dir_path=training_schema.input_folder):
        line = Preprocessor.sanitize(text=training_line)
        model.partial_fit(line)

    model.save(training_schema.trained_model)
    logging.info("The model has been trained and saved successfully.")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    main_train(file_str_path=argv[1])
