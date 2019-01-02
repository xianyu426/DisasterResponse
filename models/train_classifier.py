import sys
import nltk
nltk.download(['punkt', 'wordnet'])

import pandas as pd
from sqlalchemy import create_engine

import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

import pickle


def load_data(database_filepath):
    '''
    The function is to load the dataset from database.
    args: database file path
    return: Series, the messages, categories and categories names
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(database_filepath, engine)
    X = df['message']
    Y = df.iloc[:, 4: ]
    category_names = Y.columns.values
    return X, Y, category_names


def tokenize(text):
    '''
    The function is to process the sentence, token the words and lower it.
    arg: str text
    return:list
    '''
    word_list = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for token in word_list:
        clean_tok = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens
    


def build_model():
    '''
    The function is to build a pipeline and using gridsearch to training model.
    The pipeline including countVectorizer, TfidfTransformer to process the text and using
    RandomForestClassifier to fit the dataset
    '''
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer = tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    # set the parameters of the model
    parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
                 'tfidf__use_idf': (True, False),
                 'clf__estimator__n_estimators':[50, 100, 200],
                 'clf__estimator__max_depth':[50, 500, 1000, 5000],
                 'clf__estimator__max_features': [2000, 5000, 10000, 20000],
                 'clf__estimator__min_samples_split':[3, 5, 9]}
    cv = GridSearchCV(pipeline, param_grid = parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    The function is to return the results of prediction on test dataset, including precision socre,
    f1-score and recall score.
    args: model, test dataset and category names
    return: dict - the classification report of category names
    '''
    
    
    y_pred = model.predict(X_test) # prediction
    prediction = pd.DataFrame(y_pred.reshape(-1, 36), columns = category_names) # transform list to dataframe
    report = dict()
    for i in category_names:
        # iterate the category names and add its classification scores to dictionary 
        classification = classification_report(Y_test[i], prediction[i])
        report[i] = classification
    return report


def save_model(model, model_filepath):
    '''
    The function is to save the model using pickle.
    args: model name and file path
    return: none
    '''
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))
    
    


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
