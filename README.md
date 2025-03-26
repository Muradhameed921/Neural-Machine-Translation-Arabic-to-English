# Neural Machine Translation (NMT) - Arabic to English

This project implements a Transformer-based (Seq2Seq) Neural Machine Translation (NMT) model from scratch to perform translation from Arabic to English. The dataset used for training is the `Helsinki-NLP/tatoeba_mt` dataset. Additionally, a Streamlit web application is provided to interact with the trained model.

## Table of Contents
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Acknowledgments](#acknowledgments)
- [Results](#results)

## Dataset
The dataset used for training is loaded using the `datasets` library:
```python
from datasets import load_dataset
dataset = load_dataset('Helsinki-NLP/tatoeba_mt', 'ara-eng')
```
This dataset provides parallel sentences in Arabic and English for training the translation model.

## Model Architecture
The NMT model is based on the Transformer architecture, which consists of:
- Encoder: Processes the input Arabic sentence.
- Decoder: Generates the English translation.
- Attention Mechanism: Helps in aligning words from source to target language.

## Training
The model is trained using:
- Tokenization with SentencePiece.
- Training on a GPU-enabled environment.
- Optimizer: Adam with learning rate scheduling.
- Loss Function: Cross-entropy loss with label smoothing.

## Evaluation
The model is evaluated using:
- BLEU Score
- Translation Quality Assessment

## Deployment
A Streamlit web application is provided to interact with the trained model:
```bash
streamlit run app.py
```

## Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/yourusername/NMT-Arabic-English.git
cd NMT-Arabic-English
pip install -r requirements.txt
```

## Usage
Train the model:
```bash
python train.py
```
Run inference:
```python
python translate.py --sentence "مرحبا بك"
```
Deploy the web application:
```bash
streamlit run app.py
```

## Future Improvements
- Fine-tuning with larger datasets.
- Exploring different Transformer variants.
- Optimizing inference for real-time applications.

## Acknowledgments
- Hugging Face `datasets` library for dataset loading.
- TensorFlow/PyTorch for model implementation.
- Streamlit for easy web app deployment.

---
Feel free to contribute by opening issues or submitting pull requests!

## Results
![Arabic To English Translator](https://github.com/Muradhameed921/Neural-Machine-Translation-Arabic-to-English/blob/main/A1.jpg)
![Arabic To English Translator](https://github.com/Muradhameed921/Neural-Machine-Translation-Arabic-to-English/blob/main/A2.jpg)
![Arabic To English Translator](https://github.com/Muradhameed921/Neural-Machine-Translation-Arabic-to-English/blob/main/A3.jpg)
![Arabic To English Translator](https://github.com/Muradhameed921/Neural-Machine-Translation-Arabic-to-English/blob/main/A4.jpg)
