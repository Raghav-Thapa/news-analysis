import numpy as np
import pandas as pd
import re, string
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

import nltk
nltk.download('stopwords')

df = pd.read_csv("projectdata.csv")
words = stopwords.words("english")
words.extend(["a", "an", "the"])
words
stemmer = PorterStemmer()
df['cleaned'] = df['news_article'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x.lower()).split() if i not in words]).lower())
# df['cleaned']
df.to_csv("cleaned_projectdata.csv", index = False)
df = pd.read_csv("cleaned_projectdata.csv")
df.rename(columns={'Unnamed: 0': 'S.N'}, inplace=True)
df.to_csv("new_projectdata.csv", index = False)
df = pd.read_csv("new_projectdata.csv")
from sklearn.linear_model import LogisticRegression
log_regression = LogisticRegression()
vectorizer = TfidfVectorizer(stop_words = "english")
X = df['cleaned'] # independent
Y = df["news_category"] # dependent
X_train , X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30)


# Creating Pipeline
pipeline = Pipeline([('vect', vectorizer),
                    ("chi", SelectKBest(chi2, k=1450)),
                     ("clf", LogisticRegression(random_state = 0))])

# Training the model
model = pipeline.fit(X_train, Y_train)
# Accuracy
from sklearn.metrics import accuracy_score
predicted_category = model.predict(X_test)
print("Accuracy = ", accuracy_score(Y_test, predicted_category))
news = input("Enter news = ")
news_data = {'predict_news':[news]}
news_data_df = pd.DataFrame(news_data)
predict_news_cat = model.predict(news_data_df['predict_news'])
print("Predicted news category = ",predict_news_cat[0])



