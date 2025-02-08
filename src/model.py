from flair.data import Sentence, Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer


class Model:

    def __init__(self, train_data_folder="../train_data", test_data_folder="",
                 models_folder="../models/flair_model",  # +datetime.now().strftime("%d %m %Y %H:%M"),
                 model_path="../models/flair_model/final-model.pt", epochs=50):
        self.train_data_folder = train_data_folder
        self.test_data_folder = test_data_folder
        self.models_folder = models_folder
        self.model_path = model_path
        self.epochs = epochs

    def train(self):
        """
        train the NER model on the dataset provided in the specified folder.

        returns:
            None: The trained model is saved in the specified `models_folder`.

        """
        # Define column format (CoNLL-style)
        columns = {0: "text", 1: "ner"}

        # Load the dataset
        corpus: Corpus = ColumnCorpus(
            self.train_data_folder,
            column_format=columns,
            train_file="train.txt",
            test_file="test.txt",
            dev_file="dev.txt",
        )

        print("Korpus geladen:")
        print(corpus)

        #  Extract the tag dictionary
        label_type = "ner"
        label_dict = corpus.make_label_dictionary(label_type=label_type)
        print("Gefundene Labels:", label_dict)

        #  Define embeddings
        embedding_types_de = [
            WordEmbeddings("de"),  # Pre-trained German word embeddings
            FlairEmbeddings("de-forward"),  # German forward Flair embeddings
            FlairEmbeddings("de-backward"),  # German backward Flair embeddings
        ]
        stacked_embeddings = StackedEmbeddings(embedding_types_de)

        # 6. create Sequence-Tagger-Model
        tagger = SequenceTagger(
            hidden_size=256,
            embeddings=stacked_embeddings,
            tag_dictionary=label_dict,
            tag_type=label_type,
            use_crf=True,
        )

        # Init the trainer
        trainer = ModelTrainer(tagger, corpus)

        # Train the model
        trainer.train(
            self.models_folder,  # model save dir
            learning_rate=0.1,
            mini_batch_size=16,
            max_epochs=self.epochs,
            embeddings_storage_mode="gpu",
        )

    def evaluate(self):
        """"

        evaluate the trained NER model on the dataset provided in the specified folder.

        returns:
            None: The trained model is saved in the specified `models_folder`.
        """
        # Define column format
        columns = {0: "text", 1: "ner"}

        # Load test data
        corpus = ColumnCorpus(self.train_data_folder, column_format=columns, test_file="test.txt")

        # Load the trained model
        tagger = SequenceTagger.load(self.model_path)

        # Evaluate the model on the test set
        result = tagger.evaluate(corpus.test, gold_label_type="ner")
        print(result.detailed_results)

    def test(self, input_text):
        """
        test a trained NER model on a custom input text.

        args:
            text (str): The input text to analyze.
        """
        # load model
        tagger = SequenceTagger.load(self.model_path)

        # create sentence object
        sentence = Sentence(input_text, use_tokenizer=True)

        # apply the model to predict NER tags
        tagger.predict(sentence)

        print(sentence.get_spans("ner"))

        # show the results
        print("Erkannte Entitäten:")
        for entity in sentence.get_spans("ner"):
            print(f"Text: {entity.text}, Typ: {entity.tag}, Confidence: {entity.score:.2f}")



# using
model = Model()


# train
model.train()

# evaluate
# model.evaluate()


# # test
# input_text = "Angela Merkel besuchte Berlin am 10. Januar 2020."
# model.test(input_text)
