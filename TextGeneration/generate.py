"""Code for calling the generating a text."""

from sys import argv
import logging

from TextGeneration.utils.files import json_to_schema, schema_to_json
from TextGeneration.utils.preprocessor import Preprocessor
from TextGeneration.utils.schemas import InputSchema, OutputSchema
from TextGeneration.model import NGramsModel


def main_generate(file_str_path: str) -> None:
    """
    Call for generating a text.

    Do not modify its signature.
    You can modify the content.

    :param file_str_path: The path to the JSON that configures the generation
    :return: None
    """

    # Reading input data
    input_schema = json_to_schema(file_str_path=file_str_path, input_schema=InputSchema)

    # Load a model from a dump file.
    model = NGramsModel.load(
        trained_model=input_schema.trained_model, max_n_gram=input_schema.max_n_gram
    )
    model.use_top_candidate = input_schema.use_top_candidate

    generated_texts = []
    for input_text in input_schema.texts:
        text = Preprocessor.sanitize(text=input_text)
        generated_text = model.predict(text)

        logging.info("%s -> %s", input_text, generated_text)

        generated_texts.append(generated_text)

    # Printing generated texts
    output_schema = OutputSchema(generated_texts=generated_texts)
    schema_to_json(file_path=input_schema.output_file, schema=output_schema)


if __name__ == "__main__":
    if len(argv) > 2:
        logging.getLogger().setLevel(argv[2])

    main_generate(file_str_path=argv[1])
