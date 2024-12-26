# Fake-news-detector
#IMPLEMENTATION STEPS OF TRAINING A MODEL
          1.**DATA COLLECTION AND PREPROCESSING**
  Gather datasets:

Public Datasets:

1.FakeNewsNet: Includes real and fake news with social context.

2. Kaggle Fake News Dataser: A collection of labeled news articles.

Data Preprocessing:

Text Cleaning: Remove HTML tags, URLs, special characters, and stop words.

* Tokenization: Use Hugging Face tokenizers compatible with the chosen model.

Label Encoding: Convert labels into numerical format for model training.

* Splitting Data: Divide into training, validation, and test sets.

2. Model Development

Model Selection:

Choose a transformer model suitable for text classification 

Fine-Tuning the Model:

* Load the pre-trained model and tokenizer from Hugging Face.

* Set up a classification head on top of the transformer model.

* Define training parameters (learning rate, batch size, epochs).

* Use the training dataset to fine-tune the model.


   Why logistic regression model?
* The goal of Logistic Regression is to model the probability of a binary outcome, y, given a set of features, X.
  In this case:

Input (X): Features extracted from news articles using TF-IDF vectorization.
Output (y): Binary label indicating whether the news is real (0) or fake (1).
The model predicts a probability value : 
if val>=0.5-->label=1 -->Fake news
else-->Real news


2. The Logistic Function
Logistic Regression uses the logistic (sigmoid) function to ensure that the output is always between 0 and 1

Evaluation:

* Use the validation set to tune hyperparameters.

* Evaluate model performance on the test set using metrics like accuracy, precision, recall, and Fi-score.

* Analyze confusion matrix to understand misclassifications.




# Fake News Detection

This project aims to detect fake news articles using machine learning techniques. It involves preprocessing text data, applying Natural Language Processing (NLP) techniques, and using classification models to classify articles as real or fake.

## Features
- Text preprocessing, including stemming and stopword removal.
- Feature extraction using TF-IDF vectorization.
- Classification using machine learning models such as Logistic Regression and Decision Trees.
- Performance evaluation using metrics like accuracy and classification reports.

## Libraries Used
The following Python libraries are used in this project:
- `pandas` - For data manipulation and analysis.
- `numpy` - For numerical computations.
- `re` - For text preprocessing using regular expressions.
- `matplotlib` - For data visualization.
- `seaborn` - For enhanced data visualizations.
- `sklearn` (Scikit-learn) - For machine learning tasks, including:
  - `TfidfVectorizer` for feature extraction.
  - `train_test_split` for splitting data into training and testing sets.
  - `LogisticRegression`, `DecisionTreeClassifier`, `GradientBoostingClassifier`, and `RandomForestClassifier` for classification.
  - `accuracy_score` and `classification_report` for performance evaluation.

## Workflow
1. **Data Loading**: Load the dataset and handle missing values.
2. **Text Preprocessing**:
   - Combine relevant columns (e.g., author and title) to form the content.
   - Remove unwanted characters, lowercase the text, and remove stopwords.
   - Apply stemming to reduce words to their root forms.
3. **Feature Extraction**: Use TF-IDF vectorization to convert text data into numerical features.
4. **Data Splitting**: Split the dataset into training and testing sets.
5. **Model Training**: Train machine learning models to classify news articles.
6. **Performance Evaluation**: Evaluate models using accuracy and other metrics.

## Setup and Usage
1. Clone this repository.
2. Install the required dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
   pip install streamlit
   streamlit run app.py to view website
4. Run the Jupyter Notebook or script in collab to train and evaluate the models.
5. Use the trained model to predict whether an article is real or fake.

## Future Enhancements
- Incorporate deep learning techniques for improved accuracy.
- Add more advanced NLP techniques, such as lemmatization and named entity recognition.
- Extend the dataset for better generalization.


