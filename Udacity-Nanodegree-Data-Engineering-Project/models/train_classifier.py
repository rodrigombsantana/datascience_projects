import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle
nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + str(database_filepath))
    df = pd.read_sql_table('MSG_A_CAT', engine)
    X = df['message'].values
    Y = df.loc[:, 'related':'direct_report'].values#from related column into direct_report (last column)
    cats=list(df.columns[4:])
    
    return X, Y, cats


def tokenize(text):
    text = text.lower() #normalize
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) #normalize
    words = word_tokenize(text)#tokenize
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words] #lemmatize
    
    return lemmed

def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
    'clf__estimator__n_estimators': [50, 100],
    'clf__estimator__min_samples_split': [2, 3],
    'clf__estimator__min_samples_leaf': [10, 20],
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print(classification_report(Y_test[:,i], Y_pred[:,i]))
    



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