from goldman_emma_raw import goldman_docs
from henson_matthew_raw import henson_docs
from wu_tingfang_raw import wu_docs
# import sklearn modules here:

# Feature vectors are in the bag with scikit-learn

# checkpoint 1: import libraries
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.naive_bayes import MultinomialNB 

# Setting up the combined list of friends' writing samples
friends_docs = goldman_docs + henson_docs + wu_docs
# Setting up labels for your three friends
friends_labels = [1] * 154 + [2] * 141 + [3] * 166

# Print out a document from each friend:

mystery_postcard = """
Man issued from the womb of Mother Earth, but he knew it not, nor recognized her, to whom he owed his life. In his egotism he sought an explanation of himself in the infinite, and out of his efforts there arose the dreary doctrine that he was not related to the Earth, that she was but a temporary resting place for his scornful feet and that she held nothing for him but temptation to degrade himself. Interpreters and prophets of the infinite sprang into being, creating the "Great Beyond" and proclaiming Heaven and Hell, between which stood the poor, trembling human being, tormented by that priest-born monster, Conscience.
"""

# chackpoint 2: Create bow_vectorizer:
bow_vectorizer = CountVectorizer()

# checkpoint 3: Define friends_vectors:
friends_vectors = bow_vectorizer.fit_transform(friends_docs)

# chackpoint 4: Define mystery_vector: 
mystery_vector = bow_vectorizer.transform([mystery_postcard])

# This mystery friend gets classified(5-8)

# checkpoint 5 & 6: Define friends_classifier:
goldman_docs[39]
friends_classifier = MultinomialNB()

# checkpoint 7: Train the classifier:
friends_classifier.fit(friends_vectors, friends_labels)

# Change predictions:
predictions = friends_classifier.predict(mystery_vector)

mystery_friend = predictions[0] if predictions[0] else "someone else"

# Mystery revealed! (9-10)

# checkpoint 9: Uncomment the print statement:
print("The postcard was from {}!".format(mystery_friend))

# checkpoint 10 for Emma Goldman:
predictions = friends_classifier.predict_proba(mystery_vector)

# The postcard was from 1, which means that the classifier holds. 

