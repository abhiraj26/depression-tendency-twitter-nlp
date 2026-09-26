# Depression Tendency in Twitter Users

An NLP project that classifies tweets as suggestive or not suggestive of depression, and extends that signal to estimate the depression tendency of individual Twitter users. Full methodology and results are in `NLP_Project_Proposal_Final.pdf`.

## Notebooks

1. **BaselineNLP.ipynb** — Baseline classifier using CountVectorizer and TF-IDF features with Logistic Regression.
2. **DatasetBias.ipynb** — Analysis of dataset composition and bias, and the steps taken to reduce it.
3. **LR_GloveEmbeddings.ipynb** — Logistic Regression model built on averaged GloVe (Twitter corpus) word embeddings.
4. **NeuralModels.ipynb** — Deep learning experiments across LSTM + CNN, BiLSTM, and BiLSTM with Attention architectures.
5. **RoBERTa.ipynb** — Fine-tuned RoBERTa transformer model, trained via the SimpleTransformers library (built on Hugging Face).

## Dataset

Roughly 20K tweets were scraped from public accounts, with around 2K manually annotated as depressive and the rest supplemented with neutral/positive tweets to balance the class distribution and reduce dataset bias. Per Twitter's data-sharing policy, the raw dataset itself isn't included here, but tweet IDs can be used to reconstruct it.

## Results

Traditional baselines (Logistic Regression on TF-IDF / averaged Word2Vec) were compared against recurrent deep learning models and transformers. BiLSTM with Attention and RoBERTa were the strongest performers, with RoBERTa achieving the best overall accuracy (~92.5%).

## Requirements

- Python 3.6
- TensorFlow 2.0
- Keras 2.3
- PyTorch 1.2 / TorchVision 0.4.0
- SimpleTransformers (Hugging Face)
- Jupyter Notebook
