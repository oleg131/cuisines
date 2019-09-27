# Introduction
This is a brief example of productionizing an ML model. We will train an NLP algorithm to predict cuisine from a list of ingredients. The data used was obtain from a [Kaggle competition](https://www.kaggle.com/c/whats-cooking/data).

# Prototyping a model
As our initial approach we will choose something simple. We will lemmatize the input text using WordNet lexical database and use tf–idf for feature generation. As we will see later, this yields decent results.
```
import pandas as pd
raw = pd.read_json('train.json')
raw.head()
```
```
|    |    id | cuisine     | ingredients                                            |
|----|-------|-------------|--------------------------------------------------------|
|  0 | 10259 | greek       | ['romaine lettuce', 'black olives', 'grape tomatoe...] |
|  1 | 25693 | southern_us | ['plain flour', 'ground pepper', 'salt', 'tomatoes...] |
|  2 | 20130 | filipino    | ['eggs', 'pepper', 'salt', 'mayonaise', 'cooking o...] |
|  3 | 22213 | indian      | ['water', 'vegetable oil', 'wheat', 'salt']...]        |
|  4 | 13162 | indian      | ['black pepper', 'shallots', 'cornflour', 'cayenne...] |
```

```
import nltk
nltk.download('wordnet')

import re

from nltk.stem import WordNetLemmatizer
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

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
vectorizer = TfidfVectorizer(stop_words='english', analyzer="word")
clf = LogisticRegression(random_state=0)
estimator = make_pipeline(lemmatizer, vectorizer, clf)
```

This will create an estimator pipeline which will 
1. Lemmatize input array of array of ingredients
2. Create numerical features using tf-idf
3. Train a logistic regression classifier

Using a simple train-test split, we can assess the performance using some metric, e.g. balanced accuracy:

```
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

xtrain, xtest, ytrain, ytest = train_test_split(
    raw['ingredients'].values,
    raw['cuisine'].values, 
    test_size=0.3, random_state=0
)

estimator.fit(xtrain, ytrain)

pred = estimator.predict(xtest)

balanced_accuracy_score(ytest, pred)
> 0.6373783949355007
```

Looking at confusion matrix, we see that performance is not equally balanced across classes,

```
cm = confusion_matrix(ytest, pred)
cm = pd.DataFrame(
    cm / cm.sum(axis=1)[:, np.newaxis],
    columns=clf.classes_, index=clf.classes_
)
cm.apply(lambda x: (x * 100).round()).applymap(lambda x: int(x))
```

```
|              |   brazilian |   british |   cajun_creole |   chinese |   filipino |   french |   greek |   indian |   irish |   italian |   jamaican |   japanese |   korean |   mexican |   moroccan |   russian |   southern_us |   spanish |   thai |   vietnamese |
|--------------|-------------|-----------|----------------|-----------|------------|----------|---------|----------|---------|-----------|------------|------------|----------|-----------|------------|-----------|---------------|-----------|--------|--------------|
| brazilian    |          35 |         0 |              1 |         1 |          3 |        3 |       0 |        3 |       1 |        15 |          0 |          0 |        0 |        17 |          0 |         0 |            12 |         4 |      5 |            0 |
| british      |           0 |        33 |              1 |         0 |          0 |       17 |       0 |        2 |       3 |        13 |          0 |          0 |        0 |         2 |          0 |         0 |            26 |         0 |      0 |            0 |
| cajun_creole |           0 |         0 |             67 |         1 |          0 |        4 |       0 |        1 |       0 |         6 |          0 |          0 |        0 |         3 |          0 |         0 |            16 |         1 |      0 |            0 |
| chinese      |           0 |         0 |              0 |        89 |          0 |        1 |       0 |        0 |       0 |         2 |          0 |          2 |        1 |         1 |          0 |         0 |             1 |         0 |      2 |            1 |
| filipino     |           1 |         0 |              1 |        16 |         49 |        3 |       0 |        0 |       0 |         5 |          0 |          2 |        0 |         5 |          0 |         0 |             9 |         0 |      4 |            2 |
| french       |           0 |         1 |              1 |         0 |          0 |       63 |       1 |        0 |       1 |        23 |          0 |          0 |        0 |         1 |          0 |         1 |             8 |         1 |      0 |            0 |
| greek        |           0 |         0 |              0 |         0 |          0 |        5 |      65 |        1 |       0 |        20 |          0 |          0 |        0 |         1 |          3 |         1 |             2 |         1 |      0 |            0 |
| indian       |           0 |         0 |              0 |         0 |          0 |        0 |       1 |       93 |       0 |         1 |          0 |          0 |        0 |         2 |          0 |         0 |             1 |         0 |      1 |            0 |
| irish        |           0 |         3 |              1 |         1 |          0 |       15 |       1 |        1 |      39 |         8 |          0 |          0 |        0 |         1 |          1 |         2 |            27 |         1 |      0 |            0 |
| italian      |           0 |         0 |              0 |         0 |          0 |        4 |       1 |        0 |       0 |        91 |          0 |          0 |        0 |         1 |          0 |         0 |             2 |         0 |      0 |            0 |
| jamaican     |           1 |         1 |              2 |         3 |          2 |        2 |       0 |        3 |       0 |         5 |         63 |          0 |        0 |         4 |          0 |         2 |            11 |         1 |      0 |            0 |
| japanese     |           0 |         1 |              0 |        11 |          0 |        1 |       0 |        7 |       0 |         2 |          0 |         67 |        2 |         0 |          0 |         0 |             5 |         0 |      1 |            0 |
| korean       |           0 |         0 |              0 |        18 |          0 |        0 |       0 |        0 |       0 |         2 |          0 |          5 |       68 |         0 |          0 |         0 |             2 |         0 |      2 |            1 |
| mexican      |           0 |         0 |              0 |         0 |          0 |        1 |       0 |        0 |       0 |         2 |          0 |          0 |        0 |        93 |          0 |         0 |             3 |         0 |      0 |            0 |
| moroccan     |           0 |         0 |              2 |         0 |          0 |        2 |       1 |        8 |       0 |         7 |          0 |          0 |        0 |         2 |         72 |         0 |             3 |         2 |      0 |            0 |
| russian      |           0 |         4 |              3 |         0 |          1 |       17 |       1 |        1 |       3 |        11 |          0 |          1 |        0 |         3 |          1 |        39 |            14 |         1 |      1 |            0 |
| southern_us  |           0 |         0 |              5 |         1 |          0 |        3 |       0 |        1 |       0 |         6 |          0 |          0 |        0 |         3 |          0 |         0 |            80 |         0 |      0 |            0 |
| spanish      |           1 |         0 |              3 |         0 |          0 |       10 |       2 |        2 |       0 |        23 |          0 |          0 |        0 |         9 |          1 |         1 |             5 |        42 |      0 |            0 |
| thai         |           0 |         0 |              0 |         7 |          1 |        0 |       0 |        3 |       0 |         1 |          0 |          1 |        0 |         3 |          0 |         0 |             1 |         0 |     77 |            5 |
| vietnamese   |           0 |         0 |              0 |        13 |          2 |        0 |       0 |        2 |       0 |         1 |          0 |          2 |        1 |         1 |          0 |         0 |             1 |         0 |     28 |           48 |
```

For the last step, train estimator on the entire dataset,
```
estimator.fit(raw['ingredients'].values, raw['cuisine'].values)
```

# Storing the estimator

Now that we have trained our estimator, we need to persist it somewhere. Even though the most obvious solution is pickle, we will not be using it, because it is not robust to changes in modules and system environments. Instead we are going to extract our estimator's parameters and coefficients and store them in plain form.

```
vectorizer.idf_.tofile('idf.npy')

with open('vocabulary.json', 'w') as f:
    vocabulary = {k: int(v) for k, v in vectorizer.vocabulary_.items()}
    json.dump(vocabulary, f)

clf.coef_.tofile('coef.npy')
clf.intercept_.tofile('intercept.npy')
np.savetxt('classes.txt', clf.classes_, delimiter=',', fmt='%s')
```

In addition, we should store hyperparameters of our estimator as well, however it's not a directly serializable object, so we will just copy and paste them instead.

```
print(vectorizer.get_params())
print(clf.get_params())
```
```
{'analyzer': 'word', 'binary': False, 'decode_error': 'strict', 'dtype': <class 'numpy.float64'>, 'encoding': 'utf-8', 'input': 'content', 'lowercase': True, 'max_df': 1.0, 'max_features': None, 'min_df': 1, 'ngram_range': (1, 1), 'norm': 'l2', 'preprocessor': None, 'smooth_idf': True, 'stop_words': 'english', 'strip_accents': None, 'sublinear_tf': False, 'token_pattern': '(?u)\\b\\w\\w+\\b', 'tokenizer': None, 'use_idf': True, 'vocabulary': None}

{'C': 1.0, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 100, 'multi_class': 'warn', 'n_jobs': None, 'penalty': 'l2', 'random_state': 0, 'solver': 'warn', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
```

# Creating a web app
Now we can start creating a web app for our model using Flask. We will pretty much re-use the code from prototyping but skip the fitting part and go directly to predicting.

Our web app consists of two components. First is root `/` which will serve `index.html` as the entrypoint. Second will be mounted on `/get` which will return the actual predictions from the model.

```
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
```

For `index.html` we will use JQuery to employ AJAX to get predictions without reloading and Highcharts to display the results,

```

<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <meta name="description" content="">
    <meta name="author" content="">

    <title>ML In Production</title>

    <link rel="canonical" href="https://getbootstrap.com/docs/4.0/examples/checkout/">

    <!-- Bootstrap core CSS -->
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.9.0/css/all.min.css" integrity="sha256-UzFD2WYH2U1dQpKDjjZK72VtPeWP50NoJjd26rnAdUI=" crossorigin="anonymous" />

    <script src="https://code.jquery.com/jquery-3.4.1.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/highcharts@7.1.2/highcharts.min.js"></script>

    <script>
      $(document).ready(function() {
        var chart = Highcharts.chart("result", {
          chart: {
            inverted: true,
            backgroundColor: "#f8f9fa",
            style: {
                fontFamily: ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"]
            }
          },
          title: {
              text: "Prediction probabilities"
          },
          xAxis: {
            categories: []
          },
          series: [{
              type: "column",
              colorByPoint: true,
              data: [],
              showInLegend: false,
              name: ""
          }]
        });

        $("#submit").click(function() {
          $.get("/get", {query: $("#text").val()}, function(data) {
            chart.series[0].setData(data);
            chart.update({
              xAxis: {
                categories: data.map(x => x["name"])
              }
            })

            $("#result").show();
          })
        })
      });
    </script>

  </head>

  <body class="bg-light">

    <div class="container">
      <div class="py-5 text-center">
        <i class="fas fa-pepper-hot fa-4x fa-fw" style="padding-bottom: 1vh"></i>
        <h2>Cuisine Model</h2>
        <p class="lead">Below is an example of estimator predicting cuisine from a list of ingredients.</p>
      </div>

      <div class="row">
        <div class="col-md-12 order-md-1">
          <h4 class="mb-3">Enter ingredients</h4>
          <form class="needs-validation" novalidate>
            <div class="input-group">
              <textarea id="text" class="form-control" aria-label="With textarea" style="height: 20vh">kimchi</textarea>
            </div>
            <hr class="mb-4">
            <button id="submit" class="btn btn-primary btn-lg btn-block" type="button">Submit</button>
          </form>
        </div>
      </div>

      <hr class="mb-4">

      <div class="row">
        <div class="col-md-12">
          <div id="result" style="display: none"></div>
        </div>
      </div>

    </div>

  </body>
</html>
```

Now, if we run `python main.py` and go to `localhost:5000`, we can test our web app.

# Containerizing our web app

For each of deployment we will create a Dockerfile which contains everything necessary to make our app portable. We use `python:3.7-slim` as the base image and gunicorn as HTTP server.

```
FROM python:3.7-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

RUN python -c "import nltk; nltk.download('wordnet')"

COPY . /app

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "wsgi:app"]
```

Once we build our Docker image, we can run it with exposing its `8000` port,

```
docker build .
docker run -p 8000:8000 <image id>
```

# Adding our app to web server
Depending on which web server you use, directions will be different. Below is a sample config for NGINX,

```
server {
    listen 80;
    listen [::]:80;
    server_name server.com;
    location / {
        proxy_pass http://127.0.0.1:8000/;
    }
}
```

This tells NGINX to listen for domain `server.com` on port `80` and serve it whatever is running on port `8000` locally, i.e. our web app.
