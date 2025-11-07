# 🧠 Named Entity Recognition (NER) for Historical German Texts using Flair and spaCy  
## 🇬🇧 English Version  

### 📘 Overview  
This project implements a **Named Entity Recognition (NER)** pipeline for **historical German texts**, trained and evaluated using the **Flair** NLP framework.  
It enables the automated detection of entities such as **persons, places, organizations, and events** in archival or literary documents.  
The approach compares three systems:  
1. A **fine-tuned Flair model**,  
2. The **standard Flair model** ([flair/ner-german](https://huggingface.co/flair/ner-german)), and  
3. A **spaCy baseline model** (`de_core_news_lg`).  

---

### ⚙️ Technologies  
 -------------------------------------------------------------------------------------------------------------
| Component                |                   Description                                                    |
|--------------------------|----------------------------------------------------------------------------------|
| **Frameworks**           |        Flair NLP, spaCy, PyTorch                                                 |
| **Language Model**       |        German NER                                                                |
|                          |       (Hugging Face: [flair/ner-german](https://huggingface.co/flair/ner-german))|
| **Data Format**          |       CoNLL mit BIO-Tagging                                                      |
| **Evaluation Metrics**   |       Precision, Recall, F1, Accuracy                                            |
| **Programming Language** |       Python 3.10+                                                               |
--------------------------------------------------------------------------------------------------------------

---

### 🧩 Project Structure  

------------------------------------------------------------------------------------------------------------------------------------------
| File                          | Description                                                                                             |
|-------------------------------|-------------------------------------------------------------------------------------------------------- |
| `data_processing_flair.py`    | Converts annotated German text data into CoNLL (BIO-tagged) format for Flair.                           |
| `model.py`                    | Defines and trains a **Flair SequenceTagger** using stacked embeddings (Word + Flair forward/backward). |
| `ner_classifier.py`           | Performs entity recognition using (1) fine-tuned model, (2) standard Flair, (3) spaCy baseline.         |  
| `flair_standard.py`           | Evaluates the default Flair NER model (`de-ner-large`).                                                 |
| `evaluate_models.py`          | Compares output JSONs using Precision, Recall, F1, and Accuracy.                                        |
| `Geinitz 21774 NER.txt`       | Example annotated text with `<LOC>`, `<PER>`, `<ORG>`, `<EVENT>` tags.                                  |
| `best-model.pt_output.json`   | Output from the fine-tuned Flair model.                                                                 |
| `std_flair_model_output.json` | Output from the standard Flair model.                                                                   |
| `std_spacy_model_output.json` | Output from the spaCy model.                                                                            |
------------------------------------------------------------------------------------------------------------------------------------------

---

### 🧠 Workflow  

#### 1️⃣ Data Processing  
Annotated XML-like texts (e.g., `<PER>Leibniz</PER>`) are transformed into CoNLL-formatted BIO sequences using `DataProcessor`.  

#### 2️⃣ Model Training  
A **SequenceTagger** (BiLSTM + CRF) is trained with stacked embeddings:  
```python
embeddings = StackedEmbeddings([
    WordEmbeddings('de'),
    FlairEmbeddings('de-forward'),
    FlairEmbeddings('de-backward')
])
```

#### 3️⃣ Evaluation  
The `Evaluator` script calculates:  
- **Precision (P)**, **Recall (R)**, **F1**, and **Accuracy**  
- Comparison between Ground Truth and model outputs  
- Support for partial matches and fuzzy word alignment  

---

### 📈 Example Results  
-------------------------------------------------------------
| Model            | Precision  |  Recall | F1   | Accuracy |
|------------------|----------- |-------- |------|----------|
| Fine-tuned Flair |   0.88     |  0.86   | .87  |   0.91   |
| Standard Flair   |   0.84     |  0.81   | 0.83 |   0.88   |
| spaCy Baseline   |   0.73     |  0.69   | 0.71 |   0.80   |
-------------------------------------------------------------


The fine-tuned Flair model demonstrates clear improvements, especially in detecting **historical person names**, **archaic spellings**, and **context-specific locations**.

---
> Base Model: [Flair NER German – Hugging Face](https://huggingface.co/flair/ner-german)

---
---

## 🇩🇪 Deutsche Version  

### 📘 Übersicht  
Dieses Projekt implementiert eine **Named Entity Recognition (NER)**-Pipeline zur automatischen Erkennung von **Eigennamen in historischen deutschen Texten** mithilfe des **Flair**-Frameworks.  
Das Ziel besteht darin, Entitäten wie **Personen**, **Orte**, **Organisationen** und **Ereignisse** aus historischen Quellen oder literarischen Texten zu identifizieren.  

Verglichen werden drei Modelle:  
1. Ein **feinabgestimmtes (fine-tuned)** Flair-Modell,  
2. Das **Standardmodell von Flair** ([flair/ner-german](https://huggingface.co/flair/ner-german)),  
3. Ein **spaCy-Basismodell** (`de_core_news_lg`).  

---

### ⚙️ Technologien
 -------------------------------------------------------------------------------------------------------------
| Komponente              |                   Beschreibung                                                   |
|-------------------------|----------------------------------------------------------------------------------|
| **Frameworks**          |        Flair NLP, spaCy, PyTorch                                                 |
| **Sprachmodell**        |        Deutsches NER-Modell                                                      |
|                         |       (Hugging Face: [flair/ner-german](https://huggingface.co/flair/ner-german))|
| **Datenformat**         |       CoNLL mit BIO-Tagging                                                      |
| **Evaluationsmetriken** |       Precision, Recall, F1, Accuracy                                            |
| **Programmiersprache**  |       Python 3.10+                                                               |
--------------------------------------------------------------------------------------------------------------

---

### 🧩 Projektstruktur  
-----------------------------------------------------------------------------------------------------------------------------------
| Datei                         | Beschreibung                                                                                     |
|-------------------------------|--------------------------------------------------------------------------------------------------|
| `data_processing_flair.py`    | Konvertiert annotierte Texte in das CoNLL-Format (BIO-Tagging).                                  |
| `model.py`                    | Trainiert einen **SequenceTagger** mit gestapelten Einbettungen (Word + Flair forward/backward). |
| `ner_classifier.py`           | Führt NER mit drei Varianten aus: Fine-tuned Flair, Standard Flair, spaCy.                       |
| `flair_standard.py`           | Testet das Standardmodell `de-ner-large` von Flair.                                              |
| `evaluate_models.py`          | Bewertet Modelloutputs (Precision, Recall, F1, Accuracy).                                        |
| `Geinitz 21774 NER.txt`       | Beispielannotierter Text mit Entitätstags (`<PER>`, `<LOC>`, `<ORG>`, `<EVENT>`).                |
| `best-model.pt_output.json`   | Ergebnisse des trainierten Flair-Modells.                                                        |
| `std_flair_model_output.json` | Ergebnisse des Standard-Flair-Modells.                                                           |
| `std_spacy_model_output.json` | Ergebnisse des spaCy-Modells.                                                                    |
-----------------------------------------------------------------------------------------------------------------------------------
---

### 🧠 Verarbeitungsprozess  

#### 1️⃣ Datenaufbereitung  
Die Rohdaten im XML-ähnlichen Format werden mit `DataProcessor` in CoNLL-BIO-Form konvertiert.  

#### 2️⃣ Modelltraining  
Ein **SequenceTagger** (BiLSTM + CRF) wird mit gestapelten Einbettungen trainiert:  
```python
embeddings = StackedEmbeddings([
    WordEmbeddings('de'),
    FlairEmbeddings('de-forward'),
    FlairEmbeddings('de-backward')
])
```

#### 3️⃣ Evaluation  
Das Evaluationsskript `Evaluator` berechnet:  
- **Precision**, **Recall**, **F1** und **Accuracy**  
- Vergleich zwischen Goldstandard und Modellvorhersagen  
- Unterstützung für partielle Übereinstimmungen und Wortähnlichkeitsabgleich  

---

### 📈 Beispielergebnisse  

-------------------------------------------------------------
|     Modell       | Precision | Recall | F1     | Accuracy |
|------------------|-----------|--------|--------|----------|
| Fine-tuned Flair |   0.88    |  0.86  |  0.87  |    0.91  |
| Standard Flair   |   0.84    |  0.81  |  0.83  |   0.88   |
| spaCy Baseline   |   0.73    |  0.69  |  0.71  |   0.80   |
-------------------------------------------------------------


Das feinabgestimmte Flair-Modell zeigt deutliche Verbesserungen, insbesondere bei der Erkennung historischer **Personennamen**, **Ortsbezeichnungen** und **archaischer Schreibweisen**.

---
> Basismodell: [Flair NER German – Hugging Face](https://huggingface.co/flair/ner-german)
