LANGUAGE QUANTIFICATION

Read the News Analysis

Newspapers and their online formats supply the public with the information we need to understand the events occurring in the world around us. From politics to sports, the news keeps us informed, in the loop, and ready to make decisions about how to act in a rapidly changing world.

Given the vast amount of news articles in circulation, identifying and organizing articles by topic is a useful activity. This can help you sift through the enormous amount of information out there so you can find the news relevant to your interests, or even allow you to build a news recommendation engine!

The News International is the largest English language newspaper in Pakistan, covering local and international news across a variety of sectors. A selection of articles from a Kaggle Dataset of The News International articles is provided in the workspace.

In this project you will use term frequency-inverse document frequency (tf-idf) to analyze each article’s content and uncover the terms that best describe each article, providing quick insight into each article’s topic.

Imports and Data Preparation

1. In order to calculate tf-idf scores for the articles in the news dataset, you will need some help from scikit-learn. Begin by importing CountVectorizer, TfidfTransformer, and TfidfVectorizer from sklearn.feature_extraction.text.

2. Provided in articles.py is a selection of 10 articles from The News International. Each article, stored as a string, is given as a corpus in the list articles.

In script.py, print one of the articles and read its contents.

3. Before proceeding, let’s preprocess each article by performing tokenization and lemmatization.

Provided in preprocessing.py is a function preprocess_text() that accepts a string as input and returns a preprocessed string.

Preprocess each article in articles and store the processed articles in a list called processed_articles.

Print out one of the preprocessed articles.

Calculate Tf-idf Scores 

4. You want to begin your analysis by starting off with simple word counts for each article. Initialize a CountVectorizer object assigned to a variable named vectorizer.

5. Fit and transform your vectorizer on processed_articles to get the word counts for each article. Save the resulting counts to a variable named counts.

After you save the word counts to counts, you will see a DataFrame appear in the browser component. View the DataFrame to see the word counts for each article.

6. Now that you have the word counts for each article, let’s convert them into tf-idf scores.

Initialize a TfidfTransformer object with keyword argument norm=None saved to a variable transformer.

7. Fit and transform your transformer on counts to convert the word counts into tf-idf scores for each article. Save the resulting tf-idf scores to a variable named tfidf_scores_transformed.

After you save the tf-idf scores to tfidf_scores_transformed, you will see another DataFrame appear in the browser component. View the DataFrame to see the tf-idf scores for each article.

8. Amazing! Now you have your tf-idf scores for each article. You want to confirm, however, that the TfidfTransformer gives the same results as directly using the TfidfVectorizer.

Initialize a TfidfVectorizer object with keyword argument norm=None saved to a variable vectorizer.

9. Fit and transform your vectorizer on processed_articles to calculate the tf-idf scores for each article in one step. Save the resulting tf-idf scores to a variable named tfidf_scores.

After you save the tf-idf scores to tfidf_scores, you will see another DataFrame appear in the browser component. View the DataFrame to see the tf-idf scores for each article.

Do the tf-idf scores given by TfidfVectorizer look the same as those given by TfidfTransformer?

10. Let’s confirm that the tf-idf scores given by TfidfTransformer and TfidfVectorizer are the same.

Paste the following if statement into script.py under the comment “check if tf-idf scores are equal”:

if np.allclose(tfidf_scores_transformed.todense(), tfidf_scores.todense()):
  print(pd.DataFrame({'Are the tf-idf scores the same?':['YES']}))
else:
  print(pd.DataFrame({'Are the tf-idf scores the same?':['No, something is wrong :(']}))
You should see that the tf-idf scores are, in fact, the same!

Analyze the Results 

11. A simple way of identifying the “topic” of a document is to label the document with its highest-scoring tf-idf term. While this is a more naive approach than others, it is a quick and easy way of getting insight into the topic of a document.

Scroll down to the bottom of script.py, and begin by writing a for loop that iterates a variable i through the values 1 to 10.

12. The Pandas Series method .idxmax() is a helpful tool for returning the index of the highest value in a DataFrame column. We will use this method to find the highest scoring tf-idf term for each article.

Within the for loop, paste the following code:

print(df_tf_idf[[f'Article {i}']].idxmax())
On each pass through the for loop, this code will print the index of the term with the highest tf-idf score for that article (from Article 1 to Article 10).

Compare the actual text of the articles to the selected term. Do printed terms give you any insight into the topic of the respective articles?