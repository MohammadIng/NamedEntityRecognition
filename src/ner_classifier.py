import glob
import json
import os
import re
import spacy

from flair.data import Sentence
from flair.models import SequenceTagger
from charset_normalizer import detect
from flair.nn import Classifier


class NERClassifier:

    def __init__(self,
                 std_spacy = False,
                 std_flair = False,
                 json_data_folder="../json_data",
                 input_data_folder="../input_data",
                 model_path="../models/flair_model/",
                 model_type="final-model.pt"):

        self.json_data_folder = json_data_folder
        self.input_data_folder = input_data_folder
        self.std_spacy = False
        self.std_flair = False

        if std_spacy:
            self.output_file = os.path.join(self.json_data_folder,"std_spacy_model_output.json")
            self.std_spacy = True

        elif std_flair:
            self.output_file = os.path.join(self.json_data_folder,"std_flair_model_output.json")
            self.std_flair = True

        else:
            self.output_file = os.path.join(self.json_data_folder, model_type + "_output.json")
            self.model_type = model_type
            self.model_path = model_path + self.model_type
            self.model = SequenceTagger.load(self.model_path)  # Load the NER model directly during initialization


    @staticmethod
    def extract_meta_data(file_path):
        file_name = os.path.basename(file_path)
        match = re.match(r"([^\s]+)\s+([^\s]+)", file_name)
        sender = match.group(1) if match else "unknown"
        file_id = match.group(2).split(".")[0] if match else "unknown"

        return sender.replace("korrigiert", ""), file_id


    def read_files(self):
        """
        Find and read all .txt files in the specified folder.
        Returns:
            dict: A dictionary with file names as keys and file content as values.
        """
        txt_files = glob.glob(os.path.join(self.input_data_folder, "*.txt"))  # Find all .txt files
        files_content = {}

        for file_path in txt_files:
            with open(file_path, "rb") as file:  # Read as binary
                raw_data = file.read()
                encoding_info = detect(raw_data)
                detected_encoding = encoding_info['encoding']
                try:
                    content = raw_data.decode(detected_encoding)
                    files_content[file_path] = content
                except Exception as e:
                    print(f"Fehler beim Lesen der Datei {file_path}: {e}")
                    continue

        return files_content

    def classify_entities(self, file_path, text):
        """
        Classify NER entities in a single file's text content.
        Args:
            file_path (str): The file path of the .txt file.
            text (str): The content of the file.
        Returns:
            dict: A dictionary with the extracted information.
        """
        # extract meta data
        sender, file_id = self.extract_meta_data(file_path)

        #  NER- classify
        sentence = Sentence(text, use_tokenizer=True)
        self.model.predict(sentence)

        # extract ner
        entities = []
        for entity in sentence.get_spans("ner"):
            print("Entity:  ", entity)
            entities.append([
                entity.text,  # Entity text
                entity.tag,  # Entity label
                entity.start_position,  # Start position
                entity.end_position  # End position
            ])

        # create result format
        return {
            "id": file_id,
            "sender": sender,
            "text": text,
            "entities": entities
        }

    def classify_entities_std_spacy(self, file_path, text):
        """
        Classify NER entities in a single file's text content.
        Args:
            file_path (str): The file path of the .txt file.
            text (str): The content of the file.
        Returns:
            dict: A dictionary with the extracted information.
        """
        # extract meta data
        sender, file_id = self.extract_meta_data(file_path)

        #  NER- classify
        nlp = spacy.load("de_core_news_lg")
        doc = nlp(text)
        entities = []
        for ent in doc.ents:
            # Replace DATE with EVENT
            label = "EVENT" if ent.label_ == "DATE" else ent.label_
            entities.append([ent.text, label, ent.start_char, ent.end_char])

        # create result format
        return {
            "id": file_id,
            "sender": sender.replace("korrigiert", ""),
            "text": text,
            "entities": entities
        }

    def classify_entities_std_flair(self, file_path, text):
        """
        Classify NER entities in a single file's text content.
        Args:
            file_path (str): The file path of the .txt file.
            text (str): The content of the file.
        Returns:
            dict: A dictionary with the extracted information.
        """
        # extract meta data
        sender, file_id = self.extract_meta_data(file_path)

        #  NER- classify
        model = "de-ner"
        tagger = Classifier.load(model)
        sentence = Sentence(text)
        tagger.predict(sentence)

        entities = []
        for entity in sentence.get_spans('ner'):
            # Replace DATE with EVENT
            label = "EVENT" if entity.tag == "DATE" else entity.tag
            entities.append([entity.text, label, entity.start_position, entity.end_position])

        # create result format
        return {
            "id": file_id,
            "sender": sender.replace("korrigiert", ""),
            "text": text,
            "entities": entities
        }

    def process_all_files(self):
        """
        Process all .txt files and save results to a JSON file.
        """
        all_results = {"results": []}

        # load all files
        files_content = self.read_files()

        # classify each file
        for file_path, text in files_content.items():

            if self.std_spacy:
                result = self.classify_entities_std_spacy(file_path, text)
            elif self.std_flair:
                result = self.classify_entities_std_flair(file_path, text)
            else:
                result = self.classify_entities(file_path, text)
            all_results["results"].append(result)

        # save all results in json file
        os.makedirs(self.json_data_folder, exist_ok=True)  # Create the folder if it does not exist
        with open(self.output_file, "w", encoding="utf-8") as json_file:
            json.dump(all_results, json_file, indent=2, ensure_ascii=False)

        print(f"all result were saved in {self.output_file} .")


# using
ner = NERClassifier()
ner.process_all_files()

ner = NERClassifier(model_type="best-model.pt")
ner.process_all_files()


ner = NERClassifier(std_spacy=True)
ner.process_all_files()


ner = NERClassifier(std_flair=True)
ner.process_all_files()