from flair.data import Sentence
from flair.nn import Classifier
import os
from charset_normalizer import detect


def read_text(file_path=""):
    """
    Reads the content of a file and returns it as a string, automatically detecting the encoding.
    Args:
        file_path (str): Path to the file to be read.
    Returns:
        str: The content of the file as a string, or an empty string if the file is empty or cannot be read.
    """
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, "rb") as f_in:  # Open in binary read mode
            raw_data = f_in.read()
            encoding_info = detect(raw_data)  # Automatically detect encoding
            detected_encoding = encoding_info['encoding']

            if detected_encoding:
                try:
                    # Decode the file content with the detected encoding
                    return raw_data.decode(detected_encoding)
                except Exception as e:
                    print(f"Error decoding file: {e}")
                    return ""
            else:
                print("Encoding could not be detected.")
                return ""
    else:
        print(f"The file '{file_path}' is either empty or does not exist.")
        return ""


def classify_sentence(file_content, model="de-ner"):
    """
    Classifies the given text for Named Entity Recognition (NER) using a specified Flair model.
    Args:
        file_content (str): The input text to classify.
        model (str): The name of the Flair NER model to use (default is 'de-ner').
    """
    # Load the specified Flair model
    tagger = Classifier.load(model)

    # Create a Sentence object from the input text
    sentence = Sentence(file_content)

    # Predict NER tags for the sentence
    tagger.predict(sentence)

    # Print the sentence with predicted NER tags
    print(sentence)

    for entity in sentence.get_spans('ner'):
        print(f"{entity.text}: {entity.tag}")


# Read text from a file and classify it using the 'de-ner-large' model
text = read_text(file_path="../input_data/Krause BKW-K059-009-002-002 korrigiert.txt")
classify_sentence(file_content=text, model="de-ner-large")
