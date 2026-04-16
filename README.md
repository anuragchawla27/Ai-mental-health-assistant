🧠 Mental Health Text Classification (NLP Mini Project)

An end-to-end Machine Learning project that classifies mental health conditions from user text and provides supportive, safety-aware responses through an interactive Streamlit UI.

🚀 Overview

This project implements a mental health classification system that analyzes user-written text and predicts psychological conditions such as:

Anxiety
Depression
Stress
Bipolar
Personality Disorder
Suicidal
Normal

The system is designed with a safety-first approach, especially for detecting high-risk cases like suicidal intent.

🎯 Objective

Build a reliable NLP-based classification model
Handle class imbalance effectively
Prioritize recall for critical classes (e.g., Suicidal)
Provide real-time predictions via UI
Deliver supportive and meaningful responses
📊 Dataset
Mental Health Sentiment Dataset (~26,350 samples)
7 labeled mental health categories
Real-world user-like text inputs

⚙️ Project Workflow
Raw Text → Cleaning → Tokenization → Lemmatization → TF-IDF → Model Training → Evaluation → UI Deployment

🧹 Data Preprocessing
The following preprocessing steps were applied:

Lowercasing text
Removing special characters and noise
Tokenization (NLTK)
Stopword removal (keeping negations like "not")
Lemmatization (WordNet)
Rejoining tokens
👉 Ensures the model learns meaningful linguistic patterns instead of noise

🔍 Exploratory Data Analysis (EDA)
Checked dataset structure and null values
Analyzed class distribution
Visualized class imbalance
📌 Key Insight:
Dataset is imbalanced
Certain emotional classes dominate
🔢 Feature Engineering
Used TF-IDF Vectorization:

max_features = 10000
ngram_range = (1,2)
sublinear_tf = True
min_df = 2
max_df = 0.95
👉 Converts text into numerical form while preserving important features

✂️ Train-Test Split
80% Training
20% Testing
Used stratified sampling
🤖 Models Implemented
🔹 1. Support Vector Machine (SVM) 
Algorithm: LinearSVC
Handles high-dimensional sparse data efficiently

🔹 2. Random Forest
Ensemble-based model
Used for comparison

🔹 3. Logistic Regression
Linear-based model
Stratified splitting
Optimized using Weighted F1 Score
📈 Model Evaluation
Metrics used:

Accuracy
Precision
Recall
F1 Score
Weighted F1 Score (Primary Metric)
🧠 Why Weighted F1?
Treats all classes equally
Prevents bias toward majority classes
Crucial for detecting minority classes like Suicidal
🏆 Model Comparison

| Model                 | Accuracy | Macro F1 | Observation                                      |
|----------------------|----------|----------|--------------------------------------------------|
| Logistic Regression  | ~0.77    | ~0.77    | Best overall performance, balanced across classes |
| SVM                  | ~0.75    | ~0.77    | Stable and consistent                            |
| Random Forest        | ~0.72    | ~0.70    | Weak on minority classes                         |

🔧 Hyperparameter Tuning on SVM and Logistic Regression model
Performed using GridSearchCV:

param_grid = { 'C': [0.01, 0.1, 1, 10], 'max_iter': [3000, 5000] } ✅ Best Parameters: C = 0.1,class_weight="balanced"


| Model                         | Accuracy | F1 Score | Remark                  |
|------------------------------|----------|----------|--------------------------|
| SVM (Tuned)                  | 0.77     | 0.77     | Stable & balanced        |
| Logistic Regression (Tuned)  | ⭐ 0.78  | ⭐ 0.78  | Best overall performance |

🏆 Conclusion: Tuned Logistic Regression outperformed SVM in both accuracy and F1 score, making it the final selected model for deployment.

👉 Improved generalization and reduced overfitting

🧪 Sample Prediction
Input:

I do not feel like living anymore

Output:

Prediction: Suicidal
⚠️ Please seek immediate help
📞 Helpline (India): 9272333922 💻 UI (Streamlit App) Interactive text input Real-time prediction Color-coded outputs Safety alerts for high-risk cases
