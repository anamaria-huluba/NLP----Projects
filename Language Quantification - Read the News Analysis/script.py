import pandas as pd
import numpy as np
from articles import articles
from preprocessing import preprocess_text

# Imports and Data Preparation

# checkpoint 1: import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer 

# checkpoint 2: view article number 1
print(articles[0])

# checkpoint 3: preprocess articles
processed_articles = [preprocess_text(article) for article in articles]

# Calculate Tf-idf Scores(4-10)

# checkpoint 4: initialize and fit CountVectorizer
vectorizer = CountVectorizer()

# checkpoint 5: convert counts to tf-idf
counts = vectorizer.fit_transform(processed_articles)

# checkpoint 6: initialize and fit TfidfVectorizer
transformer = TfidfTransformer(norm=None)

# checkpoint 7: check if tf-idf scores are equal
tfidf_scores_transformed = transformer.fit_transform(counts)

# checkpoint 8: check results to the ones from TfidfTransformer 
vectorizer = TfidfVectorizer(norm=None)

# checkpoint 9: tf-idf scores given by TfidfVectorizer look the same as those given by TfidfTransformer
tfidf_scores = vectorizer.fit_transform(processed_articles) 

# checkpoint 10: confirm the scores are the same? YES
if np.allclose(tfidf_scores_transformed.todense(), tfidf_scores.todense()):
  print(pd.DataFrame({'Are the tf-idf scores the same?':['YES']}))
else:
  print(pd.DataFrame({'Are the tf-idf scores the same?':['No, something is wrong :(']}))

# Analyze the Results(11-12)

# get vocabulary of terms
try:
  feature_names = vectorizer.get_feature_names()
except:
  pass

# get article index
try:
  article_index = [f"Article {i+1}" for i in range(len(articles))]
except:
  pass

# create pandas DataFrame with word counts
try:
  df_word_counts = pd.DataFrame(counts.T.todense(), index=feature_names, columns=article_index)
  print(df_word_counts)
except:
  pass

# create pandas DataFrame(s) with tf-idf scores
try:
  df_tf_idf = pd.DataFrame(tfidf_scores_transformed.T.todense(), index=feature_names, columns=article_index)
  print(df_tf_idf)
except:
  pass

try:
  df_tf_idf = pd.DataFrame(tfidf_scores.T.todense(), index=feature_names, columns=article_index)
  print(df_tf_idf)
except:
  pass

# checkpoint 11 & 12: get highest scoring tf-idf term for each article 
for i in range(1, 10):
  print(df_tf_idf[[f'Article {i}']].idxmax())