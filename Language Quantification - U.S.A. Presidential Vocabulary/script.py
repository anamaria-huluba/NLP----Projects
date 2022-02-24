import os
import gensim
import spacy
from president_helper import read_file, process_speeches, merge_speeches, get_president_sentences, get_presidents_sentences, most_frequent_words

# Preprocessing the Data(1-5)

# checkpoint 2: get list of all speech files
files = sorted([file for file in os.listdir() if file[-4:] == '.txt'])
#print(files)

# checkpoint 3: read each speech file
speeches = [read_file(file) for file in files]

# checkpoint 4: preprocess each speech
processed_speeches = process_speeches(speeches)
# to print the first inaugural address
#print(processed_speeches[0])
# to print the first sentence in the first inaugural address
#print(processed_speeches[0][0])
# tp print the first word in the first sentence in the first inaugural address
#print(processed_speeches[0][0][0])

# checkpoint 5: merge speeches
all_sentences = merge_speeches(processed_speeches)

# All Presidents(6-9)

# checkpoint 6: view most frequently used words
most_freq_words = most_frequent_words(all_sentences)
#print(most_freq_words)

# checkpoint 7: create gensim model of all speeches
all_prez_embeddings = gensim.models.Word2Vec(all_sentences, size=96, window=5, min_count=1, workers=2, sg=1)

# checkpoint 8: view words similar to freedom
similar_to_freedom = all_prez_embeddings.most_similar("freedom", topn=20)
#print(similar_to_freedom)

# One President(10-13)

# checkpoint 10: get President Roosevelt sentences
roosevelt_sentences = get_president_sentences("franklin-d-roosevelt")
#print(roosevelt_sentences)

# checkpoint 11: view most frequently used words of Roosevelt
roosevelt_most_freq_words = most_frequent_words(roosevelt_sentences)
#print(roosevelt_most_freq_words)

# checkpoint 12: create gensim model for Roosevelt
roosevelt_embeddings = gensim.models.Word2Vec(roosevelt_sentences, size=96, window=5, min_count=1, workers=2, sg=1)

# checkpoint 13: view words similar to freedom for Roosevelt
roosevelt_similar_to_freedom = roosevelt_embeddings.most_similar("freedom", topn=20)
#print(roosevelt_similar_to_freedom)

# Selection of Presidents(14-19)

# checkpoint 14: get sentences of multiple presidents
rushmore_prez_sentences = get_presidents_sentences(["washington","jefferson","lincoln","theodore-roosevelt"])

# checkpoint 15: view most frequently used words of presidents
rushmore_most_freq_words = most_frequent_words(rushmore_prez_sentences)
#print(rushmore_most_freq_words)

# checkpoint 16: create gensim model for the presidents
rushmore_embeddings = gensim.models.Word2Vec(rushmore_prez_sentences, size=96, window=5, min_count=1, workers=2, sg=1)

# checkpoint 17: view words similar to freedom for presidents
rushmore_similar_to_freedom = rushmore_embeddings.most_similar("freedom", topn=20)
#print(rushmore_similar_to_freedom)

# checkpoint 18: view words similar to people for presidents
rushmore_similar_to_people = rushmore_embeddings.most_similar("people", topn=20)
#print(rushmore_similar_to_people)

# checkpoint 19: 

