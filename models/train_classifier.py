"""
Disaster Resoponse Project
Udacity - Data Science Nanodegree

Sample Script Execution:
> python train_classifier.py ../data/DisasterResponse.db classifier.pkl

Arguments:
    1. SQLite db path -> pre-processed data
    2. pickle file name -> to save ML model
"""

import sys
import pandas as pd
import numpy as np

import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer

from sqlalchemy import create_engine
import re

from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_recall_fscore_support, make_scorer, accuracy_score, f1_score, fbeta_score, classification_report

import pickle

import warnings
warnings.filterwarnings("ignore")

def load_data(database_filepath):
    """
    Load database filepath and return data
    
    Arguments:
        database_filepath -> path to SQLite db
    Output:
        X -> feature DataFrame
        Y -> label DataFrame
        category_names -> used for data visualization (app)
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponsetbl', con=engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    Tokenize and transform input text to clean text
    
    Arguments:
        text -> list of text messages
    Output:
        clean_tokens -> clean tokenized text
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # Remove punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    
    # lemmatization
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
    Build Model pipeline
    
    Output is a tuned model that process text messages
    and apply model for scoring.
    """

    modelp = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
        ])
    
    # hyper-parameter grid
    parameters = {#'vect__ngram_range': ((1, 1), (1, 2)),
                  #'vect__max_df': (0.75, 1.0),
                  'clf__estimator__n_estimators': (50,100)
                  }

    # create model
    model = GridSearchCV(estimator=modelp,
            param_grid=parameters,
            verbose=3,
            #n_jobs = -1,
            cv=2)

    return model

def get_results(Y_test, y_pred):
    """
    Function to generate model results in dataframe format
    
    Arguments:
        y_test -> test labels
        y_pred -> predicted labels
    """
    report = pd.DataFrame(columns=['Category', 'f_score', 'precision', 'recall'])
    num = 0
    for colnm in Y_test.columns:
        precision, recall, f_score, support = precision_recall_fscore_support(Y_test[colnm], y_pred[:,num], average='weighted')
        report.at[num+1, 'Category'] = colnm
        report.at[num+1, 'f_score'] = f_score
        report.at[num+1, 'precision'] = precision
        report.at[num+1, 'recall'] = recall
        num += 1
    print('Aggregated f_score:', report['f_score'].mean())
    print('Aggregated precision:', report['precision'].mean())
    print('Aggregated recall:', report['recall'].mean())
    print('Accuracy:', np.mean(Y_test.values == y_pred))
    return report


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model results
    
    Arguments:
        model -> Scikit ML Pipeline
        X_test -> test features
        Y_test -> test labels
        category_names -> label names
    """
    # Get results and add them to a dataframe.
    y_pred = model.predict(X_test)
    results = get_results(Y_test, y_pred)
    print(results)


def save_model(model, model_filepath):
    """
    Save Model to pickle file
        
    Arguments:
        model -> GridSearchCV or Scikit Pipeline object
        model_filepath -> destination path to save .pkl file
    
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()