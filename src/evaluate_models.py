import os
import json
import re
from datetime import datetime
from difflib import SequenceMatcher
from charset_normalizer import detect
from matplotlib import pyplot as plt


class Evaluator:

    # Reads the content of a file with automatic encoding detection.
    def read_file_with_encoding(self, file_path):
        with open(file_path, "rb") as f:
            raw_data = f.read()
            encoding_info = detect(raw_data)
            detected_encoding = encoding_info['encoding']

            if detected_encoding:
                try:
                    return raw_data.decode(detected_encoding)
                except Exception as e:
                    print(f"Error decoding file {file_path}: {e}")
                    return ""
            else:
                print(f"Could not detect encoding for file {file_path}.")
                return ""

    # Extracts sender name and document ID from the file name.
    @staticmethod
    def extract_meta_data(file_path):
        file_name = os.path.basename(file_path)
        match = re.match(r"([^\s]+)\s+([^\s]+)", file_name)
        sender = match.group(1) if match else "unknown"
        file_id = match.group(2).split(".")[0] if match else "unknown"
        return sender.replace("korrigiert", ""), file_id

    # Extracts named entities from the given annotated text.
    @staticmethod
    def extract_entities_from_text(text):
        entities = []
        offset = 0
        pattern = r"<(?P<label>[^<>]+)>(?P<text>[^<>]+)</\1>"
        for match in re.finditer(pattern, text):
            label = match.group("label")
            entity_text = match.group("text")
            start = match.start("text") - offset
            end = match.end("text") - offset
            entities.append([entity_text, label, start, end])
        return entities

    # Processes files to extract ground truth entities and save them in JSON format.
    def extract_ground_truth(self):
        output_file = "../json_data/ground_truth.json"
        results = []

        file_paths = []
        for root, dirs, files in os.walk("../evaluate_data"):
            for file in files:
                file_paths.append(os.path.join(root, file))

        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue

            sender, doc_id = self.extract_meta_data(file_path)
            content = self.read_file_with_encoding(file_path)
            entities = self.extract_entities_from_text(content)

            results.append({
                "id": doc_id,
                "sender": sender,
                "entities": entities
            })

        with open(output_file, "w", encoding="utf-8") as json_out:
            json.dump({"results": results}, json_out, indent=4, ensure_ascii=False)
        print(f"Processed data saved to {output_file}")

    # Loads JSON data from a given file.
    @staticmethod
    def load_json(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Extracts entity tuples (text, label) from JSON results.
    @staticmethod
    def extract_entities(results):
        entities = []
        for result in results:
            for entity in result["entities"]:
                word = entity[0]
                label = entity[1]
                entities.append((word, label))
        return set(entities)

    # Checks if two words are similar based on a similarity threshold.
    @staticmethod
    def is_similar(word1, word2, threshold=0.8):
        similarity = SequenceMatcher(None, word1, word2).ratio()
        return similarity >= threshold

    # Matches predicted entities with ground truth using flexible matching criteria.
    def match_flexible(self, true_entities, predicted_entities, similarity_threshold=1.0):
        true_positives = set()
        false_positives = set(predicted_entities)
        false_negatives = set()

        for true_word, true_label in true_entities:
            matched = False
            for pred_word, pred_label in predicted_entities:
                if true_label == pred_label and (
                        true_word in pred_word or pred_word in true_word or
                        self.is_similar(true_word, pred_word, similarity_threshold)
                ):
                    true_positives.add((pred_word, pred_label))
                    matched = True
                    false_positives.discard((pred_word, pred_label))
            if not matched:
                false_negatives.add((true_word, true_label))

        return {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        }

    # Calculates precision, recall, F1-score, and accuracy for evaluation.
    def calculate_metrics(self, true_entities, predicted_entities):
        results = self.match_flexible(true_entities, predicted_entities)
        tp = len(results["true_positives"])
        fp = len(results["false_positives"])
        fn = len(results["false_negatives"])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        total = tp + fp + fn
        accuracy = tp / total if total > 0 else 0.0

        return precision, recall, f1, accuracy

    # Displays the evaluation results of different NER models
    def display_results(self):
        self.extract_ground_truth()

        # Extract ground truth entities from annotated text files
        ground_truth_path = "../json_data/ground_truth.json"
        tool_final_path = "../json_data/final-model.pt_output.json"
        tool_best_path = "../json_data/best-model.pt_output.json"
        tool_std_path = "../json_data/std_flair_model_output.json"
        tool_spacy_path = "../json_data/std_spacy_model_output.json"

        # Load the ground truth and model outputs from JSON files
        ground_truth = self.load_json(ground_truth_path)["results"]
        tool_final_results = self.load_json(tool_final_path)["results"]
        tool_best_results = self.load_json(tool_best_path)["results"]
        tool_std_results = self.load_json(tool_std_path)["results"]
        tool_spacy_results = self.load_json(tool_spacy_path)["results"]

        # Extract entities from ground truth and model results
        true_entities = self.extract_entities(ground_truth)
        tool_final_entities = self.extract_entities(tool_final_results)
        tool_best_entities = self.extract_entities(tool_best_results)
        tool_std_entities = self.extract_entities(tool_std_results)
        tool_spacy_entities = self.extract_entities(tool_spacy_results)

        # Calculate evaluation metrics (Precision, Recall, F1-Score, Accuracy) for each model
        metrics = {
            "Trained Final Model": self.calculate_metrics(true_entities, tool_final_entities),
            "Trained Best Model": self.calculate_metrics(true_entities, tool_best_entities),
            "Standard Flair Model": self.calculate_metrics(true_entities, tool_std_entities),
            "Standard Spacy Model": self.calculate_metrics(true_entities, tool_spacy_entities),
        }

        # Print evaluation results for each model
        for tool, (precision, recall, f1, accuracy) in metrics.items():
            print(
                f"{tool} - Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}, Accuracy: {accuracy:.2f}")

        # plots evaluation results
        self.plot_metrics(metrics)
        self.plot_individual_metrics(metrics)

    # Plots evaluation metrics for different models.
    @staticmethod
    def plot_metrics(metrics):
        tools = list(metrics.keys())
        precision = [metrics[tool][0] * 100 for tool in tools]
        recall = [metrics[tool][1] * 100 for tool in tools]
        f1_score = [metrics[tool][2] * 100 for tool in tools]
        accuracy = [metrics[tool][3] * 100 for tool in tools]
        x = range(len(tools))

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(x, precision, label='Precision', marker='o', color='blue', linewidth=2)
        ax.scatter(x, recall, label='Recall', marker='*', color='green', linewidth=2)
        ax.scatter(x, f1_score, label='F1-Score', marker='P', color='red', linewidth=2)
        ax.scatter(x, accuracy, label='Accuracy', marker='X', color='purple', linewidth=2)

        for i, (p, r, f, a) in enumerate(zip(precision, recall, f1_score, accuracy)):
            ax.text(i + 0.1, p, f"{p:.1f}%", fontsize=10, ha='center', va='bottom', color='blue')
            ax.text(i + 0.1, r, f"{r:.1f}%", fontsize=10, ha='center', va='bottom', color='green')
            ax.text(i + 0.1, f, f"{f:.1f}%", fontsize=10, ha='center', va='bottom', color='red')
            ax.text(i + 0.1, a, f"{a:.1f}%", fontsize=10, ha='center', va='bottom', color='purple')

        ax.set_xticklabels(labels=tools, rotation=45, fontsize=12)
        ax.set_xticks(x)

        ax.set_xlabel("NER Tools", fontsize=14)
        ax.set_ylabel("Metrics (%)", fontsize=14)
        ax.set_title("Comparison of NER Tools - Precision, Recall, F1-Score, and Accuracy", fontsize=16)

        ax.legend(fontsize=12)

        plt.tight_layout()
        plt.show()

    # Plots evaluation metrics for different models individual
    @staticmethod
    def plot_individual_metrics(metrics):
        tools = list(metrics.keys())
        num_tools = len(tools)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)

        colors = ['blue', 'green', 'red', 'purple']
        markers = ['o', '*', 'P', 'X']
        labels = ['Precision', 'Recall', 'F1-Score', 'Accuracy']

        for idx, tool in enumerate(tools):
            row, col = divmod(idx, 2)  # Bestimme Zeile und Spalte (2x2-Layout)
            ax = axes[row, col]  # Hole den aktuellen Achsenbereich

            precision = metrics[tool][0] * 100
            recall = metrics[tool][1] * 100
            f1_score = metrics[tool][2] * 100
            accuracy = metrics[tool][3] * 100

            values = [precision, recall, f1_score, accuracy]
            x = range(len(values))  # Positionen auf der X-Achse

            for j, (value, color, marker) in enumerate(zip(values, colors, markers)):
                ax.scatter(j, value, color=color, marker=marker, label=labels[j], s=100)

                # Werte oberhalb der Punkte anzeigen
                ax.text(j, value + 2, f"{value:.1f}%", ha='center', fontsize=10)

            ax.set_title(f"{tool}", fontsize=14)
            ax.set_xticklabels(labels, rotation=45, fontsize=10)
            ax.set_xticks(range(len(labels)))
            ax.set_ylim(0, 110)

            ax.spines['top'].set_linewidth(2)
            ax.spines['right'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['left'].set_linewidth(2)

        for idx in range(len(tools), 4):  # Wenn weniger als 4 Tools vorhanden sind
            row, col = divmod(idx, 2)
            axes[row, col].axis('off')  # Nicht genutzte Achsen deaktivieren

        fig.text(0.04, 0.5, "Metrics (%)", va='center', rotation='vertical', fontsize=14)

        fig.suptitle("Metrics Comparison per Tool (Scatterplots)", fontsize=16)

        plt.tight_layout(rect=[0.04, 0.04, 1, 0.95])
        plt.savefig(f"plots/evaluate_result_{datetime.now().strftime("%Y_%m_%d%H_%M_%S")}.png")

        plt.show()


# Run evaluation
evaluator = Evaluator()
evaluator.display_results()
