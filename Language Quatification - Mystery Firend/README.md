LANGUAGE QUANTIFICATION

Mystery Friend

You’ve received an anonymous postcard from a friend who you haven’t seen in years. Your friend did not leave a name, but the card is definitely addressed to you. So far, you’ve narrowed your search down to three friends, based on handwriting:

Emma Goldman
Matthew Henson
TingFang Wu

But which one sent you the card?

Just like you can classify a message as spam or not spam with a spam filter, you can classify writing as related to one friend or another by building a kind of friend writing classifier. You have past writing from all three friends stored up in the variable friends_docs, which means you can use scikit-learn’s bag-of-words and Naive Bayes classifier to determine who the mystery friend is!

Feature vectors are in the bag with scikit-learn

1. Near the top of script.py, import CountVectorizer from sklearn.feature_extraction.text. Below it, import MultinomialNB from sklearn.naive_bayes.

2. Define bow_vectorizer as an implementation of CountVectorizer.

3. Use your newly minted bow_vectorizer to both fit (train) and transform (vectorize) all your friends’ writing (stored in the variable friends_docs). Save the resulting vector object as friends_vectors.

4. Create a new variable mystery_vector. Assign to it the vectorized form of [mystery_postcard] using the vectorizer’s .transform() method.

(mystery_postcard is a string, while the vectorizer expects a list as an argument.)

This mystery friend gets classified

5. You’ve vectorized and prepared all the documents. Let’s take a looks at your friends’ writing samples to get a sense of how they write.

Print out one document of each friend’s writing — try any one between 0 and 140. (Your friend’s documents are stored in goldman_docs, henson_docs, and wu_docs.)

6. Have an inkling about which friend wrote the mystery card? We can use a classifier to confirm those suspicions…

Implement a Naive Bayes classifier using MultinomialNB. Save the result to friends_classifier.

7. Train friends_classifier on friends_vectors and friends_labels using the classifier’s.fit() method.

8. Change predictions value from ["None Yet"] to the classifier’s prediction about which friend wrote the postcard. You can do this by calling the classifier’s .predict() method on the mystery_vector.

Mystery revealed!

9. Uncomment the final print statement and save your code to see who your mystery friend was all along!

10. But does it really work? Find some lines by Emma Goldman, Matthew Henson, and TingFang Wu on gutenberg.org and save them to mystery_postcard to see how the classifier holds up!

Try using the .predict_proba() method instead of .predict() and print out predictions to see the estimated probabilities that the mystery_postcard was written by each person.

What happens when you add in a recent email or text instead?