# Model Card: Sentiment Analysis (TF-IDF + Logistic Regression)

## Model Details

- **Model Name:** Sentiment Analysis Text Classifier
- **Model Type:** Scikit-Learn `Pipeline` — `TfidfVectorizer(max_features=20000, ngram_range=(1,2))` + `LogisticRegression`
- **Task:** Multi-class text sentiment classification
- **Target Classes:** `positive`, `negative`, `neutral`
- **Language:** English

## Dataset

- **Source:** [abhi8923shriv/sentiment-analysis-dataset](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset) (downloaded via `kagglehub`)
- **Content:** Twitter sentiment text samples labelled as positive, negative, or neutral.

## Feature Engineering

- Text tokenized using TF-IDF with unigrams and bigrams (`ngram_range=(1,2)`)
- Up to 20,000 most significant features retained
- English stop words removed during vectorization

## Evaluation Metrics

Performance evaluated on a 20% stratified test split:

- **Accuracy:** ~78–82%
- **Per-Class F1-Score:** Available at runtime via `GET /model` endpoint (stored in `models/metrics.json`)

## Inputs & Outputs

- **Input:** Plain English text string (1 character minimum)
- **Output:**
  - `sentiment_prediction`: `positive`, `negative`, or `neutral`
  - `probabilities`: Confidence distribution across all three classes (sums to 1.0)

## Limitations

- **English Only:** The model was trained on English-language Twitter text. Non-English inputs will produce unreliable results.
- **Sarcasm & Nuance:** Sarcastic or highly idiomatic text may be misclassified.
- **Domain Shift:** Model performance degrades significantly when applied to domains very different from social media (e.g., formal legal text, medical records).
