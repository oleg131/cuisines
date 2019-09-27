import numpy as np
import re
import json

from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from flask import Flask, request, jsonify, send_from_directory
from sklearn.base import TransformerMixin

app = Flask(__name__)

class Lemmatizer(TransformerMixin):
    """Lemmatizer written using fit-predict paradigm."""
    
    def __init__(self, lemmatizer):
        self.lemmatizer = lemmatizer
        
    def lemmatize(self, x):
        """Lemmatize a single array of words."""
        
        # Join array as a single string
        x = ' '.join(x)
        
        # Remove non-letter characters
        x = re.sub('[^A-Za-z]', ' ', x)
        
        # Lemmatizer using specified lemmatizer
        x = self.lemmatizer.lemmatize(x)
        
        return x
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        return [self.lemmatize(i) for i in X]

        
lemmatizer = Lemmatizer(WordNetLemmatizer())

vectorizer = TfidfVectorizer(**{
    'analyzer': 'word',
    'binary': False,
    'decode_error': 'strict',
    'dtype': np.float64,
    'encoding': 'utf-8',
    'input': 'content',
    'lowercase': True,
    'max_df': 1.0,
    'max_features': None,
    'min_df': 1,
    'ngram_range': (1, 1),
    'norm': 'l2',
    'preprocessor': None,
    'smooth_idf': True,
    'stop_words': 'english',
    'strip_accents': None,
    'sublinear_tf': False,
    'token_pattern': '(?u)\\b\\w\\w+\\b',
    'tokenizer': None,
    'use_idf': True,
    'vocabulary': None
})

vectorizer.idf_ = np.fromfile('idf.npy')
with open('vocabulary.json') as f:
    vectorizer.vocabulary_ = json.load(f)

n_features = len(vectorizer.idf_)

clf = LogisticRegression(**{
    'C': 1.0,
    'class_weight': None,
    'dual': False,
    'fit_intercept': True,
    'intercept_scaling': 1,
    'max_iter': 100,
    'multi_class': 'warn',
    'n_jobs': None,
    'penalty': 'l2',
    'random_state': 0,
    'solver': 'warn',
    'tol': 0.0001,
    'verbose': 0,
    'warm_start': False
})

clf.coef_ = np.fromfile('coef.npy').reshape(-1, n_features)
clf.intercept_ = np.fromfile('intercept.npy')
clf.classes_ = np.loadtxt('classes.txt', delimiter=',', dtype=str)

estimator = make_pipeline(lemmatizer, vectorizer, clf)


def predict(query):
    pred = estimator.predict_proba([[query]])
    pred = {j: i for i, j in zip(pred[0], clf.classes_)}

    return pred


@app.route('/get')
def get():
    query = request.args.get('query', '')
    pred = predict(query)
    pred = [{'name': k, 'y': v} for k, v in pred.items()]
    pred = sorted(pred, key=lambda x: x['y'], reverse=True)

    return jsonify(pred)


@app.route('/')
def index():
    return send_from_directory('./', 'index.html')


if __name__ == '__main__':
    app.run(debug=True)
