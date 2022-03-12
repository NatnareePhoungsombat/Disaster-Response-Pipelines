import sys
import nltk
import re
import sqlite3
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV



def load_data(database_filepath):
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM disaster_response",engine) 
    df = df[(df["related"] == 1) & (df['genre'] != 'social')]
    
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'],  axis=1)
    category_names = Y.columns.values
    
    return X, Y, category_names



def tokenize(text):
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens



def build_model():
    
    pipeline = Pipeline([('cvect', CountVectorizer(tokenizer = tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', RandomForestClassifier())])
    
    parameters = {
        'clf__max_depth': [5, None],
        'clf__min_samples_split': [2, 3],
        'clf__n_estimators': [5, 10, 50]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return pipeline
    


def evaluate_model(model, X_test, Y_test, category_names):
    
    y_pred_test = model.predict(X_test)
    
    print('Accuracy:', accuracy_score(Y_test, y_pred_test))
    print('precision_weighted:', precision_score(Y_test, y_pred_test, average='weighted'))
    print('recall_micro:', recall_score(Y_test, y_pred_test, average='micro'))
    print('f1_macro:', f1_score(Y_test, y_pred_test, average='macro'))
    print(classification_report(Y_test, y_pred_test, target_names=category_names))



def save_model(model, model_filepath):
    
    with open(model_filepath, 'wb') as file:  
        pickle.dump(model, file)



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