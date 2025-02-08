import os
import random

from charset_normalizer import detect
from transformers import DataProcessor


class DataProcessor:

    def __init__(self):
        self.output_file = "../train_data/train.txt"
        self.base_data_folder = "../base_data"
        self.train_data_folder = "train_data"

    def read(self, file_path):
        """
        Read the content of a text file with automatic encoding detection.
        Args:
            file_path (str): Path to the text file.
        Returns:
            str: The content of the file as a string.
        """
        try:
            with open(file_path, "rb") as file:  # Read as binary
                raw_data = file.read()
                encoding_info = detect(raw_data)
                detected_encoding = encoding_info['encoding']
            content = raw_data.decode(detected_encoding)
            print("File content loaded successfully.")
            return content
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.")
            return ""
        except Exception as e:
            print(f"Error: {e}")
            return ""

    def parse_annotated_text(self, input_text):
        """
        Parse annotated text (with <TAG>...</TAG>) into CoNLL format using BIO tagging.
        Args:
            input_text (str): The input text with annotations.
        Returns:
            list: A list of (word, tag) tuples in CoNLL format.
        """
        sentences = []
        current_sentence = []
        current_tag = None  # Stores the current tag, e.g., "LOC", "EVENT"
        inside_tag = False  # Tracks if we are inside a tag

        word_buffer = []  # Temporary buffer for the current word

        i = 0
        while i < len(input_text):
            char = input_text[i]

            if char == "<":  # Start of a tag
                # Find the closing '>'
                closing_index = input_text.find(">", i)
                tag_content = input_text[i + 1: closing_index]

                if tag_content.startswith("/"):  # Closing tag
                    if word_buffer:
                        word = "".join(word_buffer).strip()
                        if word:
                            if inside_tag:
                                current_sentence.append((word, f"I-{current_tag}"))
                            else:
                                current_sentence.append((word, "O"))
                        word_buffer = []
                    inside_tag = False  # End the current tag
                    current_tag = None
                else:  # Opening tag
                    current_tag = tag_content  # Set the current tag
                    inside_tag = True
                i = closing_index  # Move index to the end of the tag
            elif char in {" ", "\n", ".", "!", "?"}:  # Word boundary or sentence end
                if word_buffer:
                    word = "".join(word_buffer).strip()
                    if word:
                        if inside_tag:
                            # Use `B-` if it’s the first word in the tag, otherwise `I-`
                            tag = f"B-{current_tag}" if not any(
                                w[1].startswith("B-") for w in current_sentence) else f"I-{current_tag}"
                            current_sentence.append((word, tag))
                        else:
                            current_sentence.append((word, "O"))
                    word_buffer = []  # Clear buffer

                if char in {".", "!", "?"}:  # Sentence end
                    current_sentence.append((char, "O"))
                    sentences.append(current_sentence)
                    current_sentence = []
            else:  # Part of a word
                word_buffer.append(char)

            i += 1

        # Add any remaining word
        if word_buffer:
            word = "".join(word_buffer).strip()
            if inside_tag:
                tag = f"B-{current_tag}" if not any(
                    w[1].startswith("B-") for w in current_sentence) else f"I-{current_tag}"
                current_sentence.append((word, tag))
            else:
                current_sentence.append((word, "O"))
        if current_sentence:
            sentences.append(current_sentence)

        return sentences

    def write_to_conll(self, sentences, output_file):
        """
        Write sentences in CoNLL format to a file.
        Args:
            sentences (list): A list of sentences, each containing (word, tag) tuples.
            output_file (str): Path to the output file.
        """
        with open(output_file, "a", encoding="utf-8") as f:
            for sentence in sentences:
                for word, tag in sentence:
                    f.write(f"{word} {tag}\n")
                f.write("\n")  # Separate sentences with an empty line

    def process_text_file(self, file_path):
        """
        Read, parse, and write annotated text into CoNLL format.
        Args:
            file_path (str): Path to the annotated text file.
            output_file (str): Path to save the CoNLL-formatted text.
        """
        text = self.read(file_path)
        if text:
            sentences = self.parse_annotated_text(text)
            self.write_to_conll(sentences, self.output_file)
            print(f"CoNLL-formatted data saved to '{self.output_file}'.")

    def split_train_file(self, test_ratio=0.20, dev_ratio=0.1, seed=42):
        """
        Split train.txt into train, test, and dev files.
        Args:
            train_file (str): Path to the original train.txt file.
            output_folder (str): Path to the folder to save split files.
            test_ratio (float): Fraction of sentences to use as test data.
            dev_ratio (float): Fraction of sentences to use as dev data.
        """
        with open(self.output_file, "r", encoding="utf-8") as f:
            sentences = f.read().strip().split("\n\n")  # Split by sentences

        random.seed(seed)
        random.shuffle(sentences)  # Shuffle the sentences

        test_size = int(len(sentences) * test_ratio)
        dev_size = int(len(sentences) * dev_ratio)

        test_sentences = sentences[:test_size]
        dev_sentences = sentences[test_size:test_size + dev_size]
        train_sentences = sentences[test_size + dev_size:]

        # Save split files
        with open(f"{self.train_data_folder}/train.txt", "w", encoding="utf-8") as f:
            f.write("\n\n".join(train_sentences))

        with open(f"{self.train_data_folder}/test.txt", "w", encoding="utf-8") as f:
            f.write("\n\n".join(test_sentences))

        with open(f"{self.train_data_folder}/dev.txt", "w", encoding="utf-8") as f:
            f.write("\n\n".join(dev_sentences))

        print("data file was split")

    def delete_train_data(self):
        file_path = "train.txt"  # Pfad zur Datei

        # Öffne die Datei im Schreibmodus und überschreibe sie mit nichts
        with open(self.output_file, "w", encoding="utf-8") as file:
            pass  # Die Datei bleibt leer

        print(f"Der Inhalt von '{file_path}' wurde gelöscht.")


    def run(self):
        # Run the process with the example text
        for file_name in os.listdir(self.base_data_folder):
            file_path = os.path.join(self.base_data_folder, file_name)
            if os.path.isfile(file_path):
                self.process_text_file(file_path)


data_processor = DataProcessor()

# delete train data
data_processor.delete_train_data()

# train data format
data_processor.run()

# split
data_processor.split_train_file()
