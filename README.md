# SENTIMENT ANALYSIS PROJECT – U.S. SENATORS’ TWEETS (May–Oct 2020)  

## Abstract

This project conducts a comparative sentiment analysis of tweets from U.S. state governors and similar state-level officials during a politically volatile period from May 1 to October 31, 2020. Utilizing natural language processing (NLP) techniques, the study classifies tweet sentiments as positive, neutral, or negative across dictionary-based, transformer-based, and supervised machine learning methods. The analysis reveals methodological differences, partisan variations in sentiment expression, temporal trends, and the relationship between sentiment and engagement metrics. Findings highlight the superiority of context-aware models for political text and provide insights into communication strategies, with applications in media monitoring and computational social science. Limitations, including the absence of human-annotated ground truth, are discussed, along with recommendations for future research.

## 1. Introduction

### 1.1 Project Topic and Rationale

In contemporary political discourse, social media platforms such as Twitter (now X) serve as critical channels for public officials to disseminate policy announcements, respond to current events, and engage constituents. This project examines sentiment in tweets from 46 U.S. state governors and similar state-level officials, focusing on a dataset comprising 30,490 tweets.

From a computational linguistics and NLP perspective, political tweets pose unique challenges due to their brevity, contextual density, and prevalence of informal elements such as hashtags, mentions, and slang. These characteristics make them an ideal benchmark for evaluating sentiment analysis models' robustness.

From an applied or "business" standpoint, such analyses inform media monitoring, reputation management, and strategic communication by identifying resonant messaging styles and engagement drivers. The selected period—encompassing legislative activities, the COVID-19 pandemic, social justice movements, and pre-election campaigning—captures heightened political volatility, enhancing the dataset's relevance.

### 1.2 Research Questions

The study addresses the following research questions:

- **RQ1 (The Partisan Divide):** To what extent do sentiment expressions (positive, neutral, or negative) in tweets from Republican and Democratic officials differ, and what do these differences reveal about their communication strategies?
- **RQ2 (Temporal Dynamics):** How do sentiment trends evolve from May to October 2020, and to what degree are these fluctuations associated with external factors such as legislative calendars and campaigning intensity?
- **RQ3 (Engagement Correlates):** Does sentiment polarity significantly correlate with engagement metrics (e.g., likes and retweets), or are other variables (e.g., account prominence and topical relevance) more predictive of virality?

## 2. Data Description

### 2.1 Primary Dataset

- **File:** `senator_twitter_May-Oct.csv` 
- **Source:** Harvard Dataverse (DOI: [10.7910/DVN/K4XSYC](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/K4XSYC))
- **Time Period:** May 1 to October 31, 2020
- **Scope:** 30,490 tweets from 46 unique political accounts, including metadata on party affiliation and state.
- **Key Variables:** Tweet text, posting date/time, engagement metrics (likes, retweets, replies), account metadata (name, Twitter ID, state, party).

### 2.2 Derived Outputs

For reproducibility, the project generates labeled datasets:
- `senator_tweets_with_sentiment_plus_vader.csv` (Dictionary-based results)
- `senator_tweets_with_sentiment.csv` (Transformer results)
- `senator_tweets_with_lr_tfidf.csv` (Supervised ML results)

## 3. Methods

The analysis employs three methodological families to classify sentiments, enabling a comparative evaluation of interpretability, contextual sensitivity, and efficiency. Preprocessing varies by method to optimize performance.

### 3.1 Dictionary- and Corpus-Based Baselines

#### 3.1.1 VADER (Valence Aware Dictionary and sEntiment Reasoner)

- **Approach:** Rule-based lexicon that scores text via word valence, incorporating heuristics for punctuation, capitalization, and negation.
- **Preprocessing:** Minimal, preserving sentiment cues like emojis and punctuation.
- **Output:** Compound score mapped to Negative (< -0.05), Neutral, or Positive (> 0.05).

#### 3.1.2 Custom Corpus Lexicon (Seed + Pointwise Mutual Information)

- **Approach:** Begins with a seed lexicon of positive/negative terms, expanded using PMI to capture corpus-specific co-occurrences.
- **Preprocessing:** Tokenization focused on dataset vocabulary.
- **Output:** Raw and length-normalized scores, thresholded into three classes.
- **Rationale:** Serves as a domain-adapted baseline, though susceptible to topic biases (e.g., political hashtags).

### 3.2 Transformer Model

- **Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest` (RoBERTa variant pre-trained on 58 million tweets).
- **Approach:** Contextual deep learning for probability distributions over Negative, Neutral, and Positive.
- **Preprocessing:** Light tokenization, removing extraneous noise while retaining context.
- **Output:** Class probabilities, confidence scores, and final labels; used to generate "silver labels" for supervised training.

### 3.3 Supervised Classical Machine Learning

- **Pipeline:** TF-IDF vectorization followed by Logistic Regression.
- **Approach:** Trained on silver labels from the transformer model; hyperparameters optimized via GridSearchCV (e.g., n-gram range, regularization strength C, class weights).
- **Preprocessing:** Normalization including lowercasing, removal of URLs, mentions, and stopwords.
- **Evaluation:** Train-test split assessing agreement with silver labels (accuracy, macro F1).

All methods are implemented in Python using libraries such as Pandas, Scikit-Learn, and Hugging Face Transformers. Code is organized into three Jupyter notebooks.

## 4. Results

### 4.1 Comparative Sentiment Distributions

The methods yield distinct distributions, underscoring biases in lexicon approaches:

| Method                  | Negative (%) | Neutral (%) | Positive (%) |
|-------------------------|--------------|-------------|--------------|
| VADER                   | 15.7        | 20.0       | 64.7        |
| Custom Corpus Lexicon   | 3.2         | 12.3       | 84.5        |
| Transformer (RoBERTa)   | 10.2        | 49.6       | 40.2        |
| TF-IDF + Logistic Regression | 10.8   | 49.2       | 40.5        |

Lexicon methods exhibit positivity bias from slogan misinterpretation, while transformers emphasize neutrality in informational content.

### 4.2 Model Agreement and Performance

- Transformer vs. Logistic Regression: ~98.7% agreement (due to silver label training).
- VADER vs. Transformer: ~58.5% agreement, highlighting contextual differences.
- Supervised Metrics (vs. silver labels): Accuracy = 81.22%, Macro F1 = 78.2%.

### 4.3 Pattern Analysis and Answers to Research Questions

- **RQ1 (Partisan Divide):** Republicans exhibit higher positive sentiment (46.5%) and lower negative sentiment (6.5%) compared to Democrats (35.9% positive, 13.8% negative), consistent across methods. This suggests Republicans employ optimistic rhetoric for support mobilization, while Democrats emphasize critiques.

- **RQ2 (Temporal Dynamics):** Sentiment varies monthly, dipping in June–July (summer lull) and peaking in September (pre-election intensity), aligning with legislative and campaigning cycles.

- **RQ3 (Engagement Myth):** Sentiment shows near-zero correlations with engagement (~0.00 for retweets, ~0.07 for likes), indicating virality is driven by account size and topic relevance rather than emotional tone.

## 5. Discussion

### 5.1 Methodological Insights

Transformers outperform lexicons by capturing context, avoiding optimism bias in slogan-heavy text. Knowledge distillation via silver labels enables efficient classical models, facilitating scalable applications. Lexicons, while interpretable, are inadequate for nuanced political discourse.

### 5.2 Implications

The partisan positivity gap reflects strategic differences, not inherent dispositions. Temporal fluctuations underscore event-driven sentiment, while weak engagement links suggest focusing on content relevance in communication strategies.

### 5.3 Value

**Scientific:** Benchmarks NLP methods on real-world political data, emphasizing evaluation without ground truth.  
**Applied:** Supports monitoring tonal shifts for reputation management and agile responses to public perception.

## 6. Limitations

- Absence of human ground truth: Evaluations measure model agreement, not absolute accuracy.
- Handling nuance: Models struggle with sarcasm, irony, and link-dependent context.
- Topic confounding: Sentiment may conflate with subject matter (e.g., crises vs. celebrations), influencing engagement independently.
- Lexicon biases: Over-sensitivity to domain-specific terms like hashtags.

Future work could incorporate human annotations, sarcasm detection, and event correlation.

## 7. Reproducibility

### 7.1 Requirements

- Python environment with Pandas, Scikit-Learn, Hugging Face Transformers, VADER.
- Download primary dataset from Harvard Dataverse.

### 7.2 Execution Guide

1. Place `senator_twitter_May-Oct.csv` in the working directory.
2. Run notebooks sequentially:
   - `senator_sentment_dictionary.ipynb` (Dictionary methods)
   - `senator_sentiment_transformers.ipynb` (Transformer)
   - `senator_sentiment_TF-IDF+LR+CV.ipynb` (Supervised ML)

## References

- Hutto, C. J., & Gilbert, E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. *ICWSM*.
- Barbosa, L., & Feng, J. (2010). Robust Sentiment Detection on Twitter from Biased and Noisy Data. *COLING*.
- Lourentzou, I., et al. (2021). cardiffnlp/twitter-roberta-base-sentiment-latest. Hugging Face Model Hub.
- Dataset: Senator Twitter Data. Harvard Dataverse, DOI: 10.7910/DVN/K4XSYC.