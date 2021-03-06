LANGUAGE QUANTIFICATION

U.S.A. Presidential Vocabulary

Whenever a United States of America president is elected or re-elected, an inauguration ceremony takes place to mark the beginning of the president’s term. During the ceremony, the president gives an inaugural address to the nation, dictating the tone and focus of the next four years of leadership.

In this project you will have the chance to analyze the inaugural addresses of the presidents of the United States of America, as collected by the Natural Language Toolkit, using word embeddings.

By training sets of word embeddings on subsets of inaugural address versus the collection of presidents as a whole, we can learn about the different ways in which the presidents use language to convey their agenda.

Preprocessing the Data

1. Provided in the workspace are .txt files for each of the 58 U.S.A. presidential inaugural addresses from 1789 to 2017. The files are named with the following convention:

YEAR-PRESIDENT.txt

where YEAR is the year the inaugural address was given and PRESIDENT is the name of the president who gave the address. If there is only one president with a given last name, only the last name is used. For presidents with the same last name, first name and middle initial(s), separated by a -, are included.

Open the file navigator and click on a few of the files to view them.

2. Navigate back to script.py. At the top of the file we have imported some libraries that will be helpful for your analysis, as well as some helper functions from the file president_helper.py.

In order to create word embeddings on the corpus of all the presidents’ speeches, we need to read the text data from each file, separate the files into sentences on a word by word basis, and then merge all the sentences across the speeches into one big list of lists.

Let’s start by finding all the file names for the .txt files we will be analyzing. Paste the following code into script.py to store a list of all .txt files in a variable files.

files = sorted([file for file in os.listdir() if file[-4:] == '.txt'])
Print files to view the names of each inauguration speech file.

3. Imported from president_helper.py is a function called read_file() which takes a file name as an argument and returns the opened file as one big string.

Call read_file() on each file in files, and save all the resulting string versions to a list named speeches.

4. Now that we have a list of speeches as strings, we need to breakdown the speeches into words on a sentence by sentence basis.

For example, if we have the following sentences:

"The cat in the hat is sad. He didn't have a mouse."
The processed data would look as follows:

[["the","cat","in","the","hat","is","sad"],["he","didn't","have","a","mouse"]]
We’ve imported the function process_speeches() from president_helper.py in order to perform this preprocessing for you. The function takes a list of strings as an argument and returns a list of lists. Each inner list represents one inaugural address and is a list of lists as well. Each inner list of the inaugural address list represents a sentence of that address, and each item in the sentence list is a word token in that sentence (see the hint for further explanation of this structure).

Call process_speeches() with speeches as an argument and save the resulting list to processed_speeches.

5. In order to build a custom set of word embeddings using gensim, we need to convert our data into a list of lists, where each inner list is a sentence and each item in the inner list is a word token.

Currently, our speech data is in the following form:

speeches = [[["monkeys","eat","bananas"],["they","hang","from","trees"],...], [["penguins","are","cute"],["they","cannot","fly"],...],...]
Where we have two speeches, one about monkeys and another about penguins. We want to remove the out layer of list so our data is as so:

speeches = [["monkeys","eat","bananas"],["they","hang","from","trees"],..., ["penguins","are","cute"],["they","cannot","fly"],...]
We’ve imported a function merge_speeches() from president_helper.py that takes a list of all our processed speeches and returns a list of lists where each inner list is a sentence and each item in the inner list is a word token.

Call merge_speeches() on processed_speeches and save the result to all_sentences.

Our data is now processed and ready for some analysis!

All Presidents

6.To get a better understanding of the data, let’s take a look at the most frequently used words across all the inaugural addresses.

Imported from president_helper.py is a function most_frequent_words(), which uses the Counter() function from Python’s collections module to find the most common words in a list of lists, where each item in the inner list is a word token.

Call most_frequent_words() with all_sentences as an argument and save the result to most_freq_words. Print most_freq_words to the terminal.

Once you have taken a look at the list of the commonly used words, you can comment out your print statement so your output terminal doesn’t display the words each time you run your code.

7. Finally, it’s word embedding time! Create a word embedding model with gensim using the following function and keyword arguments:

gensim.models.Word2Vec(__________, size=96, window=5, min_count=1, workers=2, sg=1)
Save the model to a variable named all_prez_embeddings.

Replace the __________ with your variable containing all the inaugural address sentences.

8. Now that we have our word embeddings, let’s have some fun exploring them! The concept of “freedom” is prevalent in the speeches made by the presidents. Find the top 20 words that are used in similar contexts to “freedom”, and save the results to a variable named similar_to_freedom.

Print similar_to_freedom to the terminal.

9. What other words in the corpus of inaugural addresses do you want to analyze? Pick a word from most_freq_words and find other words that are used similarly. Are you surprised by the words that are used in the same contexts?

One President

10. A fun aspect of word embeddings is to see how different corpora result in different word embeddings, alluding to differences in how words are used between writers/authors/speakers.

Let’s train a word embedding model on a single president and see how their word embeddings differ from the collection of all presidents.

Provided in script.py is a function get_president_sentences() that takes a president’s name as a string argument and returns a list of processed sentences from every inaugural address given by the president.

Call get_president_sentences() with "franklin-d-roosevelt" as an argument and save the result to roosevelt_sentences.

11. To get a better understanding of President Franklin D Roosevelt’s speeches, let’s take a look at the most frequently used words across his inaugural addresses.

Call most_frequent_words() with roosevelt_sentences as an argument and save the result to roosevelt_most_freq_words. Print roosevelt_most_freq_words to the terminal.

Once you have taken a look at the list of the commonly used words, you can comment out your print statement so your output terminal doesn’t display the words each time you run your code.

12. Now that we have the sentences from President Roosevelt’s speeches, create another word embedding model with gensim, using the same keyword arguments as in the earlier word embedding model you created, trained on just the inaugural address sentences for President Roosevelt. Save the embeddings to a variable named roosevelt_embeddings.

13. Like with our previous word embedding model, let’s explore roosevelt_embeddings! Find the top 20 words that are used in similar contexts to “freedom”, and save the results to a variable named roosevelt_similar_to_freedom.

Print roosevelt_similar_to_freedom to the terminal.

How do the words similar to “freedom” in President Roosevelt’s embeddings compare to the words similar to “freedom” in the embeddings of all the presidents’ speeches?

Are there any surprises?

Selection of Presidents

14. You may have noticed that the results from roosevelt_embeddings.most_similar("freedom", topn=20) are less than satisfying. This is because we were working with a limited dataset, producing less robust and generalizable word embeddings.

Let’s increase our corpus size and find more defined word embeddings by training a word embedding model on the inaugural addresses of a collection of multiple presidents.

Provided in script.py is a function get_presidents_sentences() that takes a list of multiple presidents’ names as an argument and returns a list of processed sentences from every inaugural address given by the group of presidents.

Call get_presidents_sentences() with ["washington","jefferson","lincoln","theodore-roosevelt"] as an argument and save the result to rushmore_prez_sentences.

15. To get a better understanding of President Washington, Jefferson, Lincoln, and Theodore Roosevelt’s speeches, let’s take a look at the most frequently used words across their inaugural addresses.

Call most_frequent_words() with rushmore_prez_sentences as an argument and save the result to rushmore_most_freq_words. Print rushmore_most_freq_words to the terminal.

Once you have taken a look at the list of the commonly used words, you can comment out your print statement so your output terminal doesn’t display the words each time you run your code.

16. Now that we have the sentences from the presidents featured on Mount Rushmore, create another word embedding model with gensim, using the same keyword arguments as in the earlier word embedding model you created, trained on just the inaugural address sentences for these presidents. Save the embeddings to a variable named rushmore_embeddings.

17. Like with our previous word embedding models, let’s explore rushmore_embeddings! Find words that are used in similar contexts to “freedom”, and save the results to a variable named rushmore_similar_to_freedom.

How do the words similar to “freedom” in these presidents’ embeddings compare to the words similar to “freedom” in the embeddings of all the presidents’ speeches? And to just Franklin D Roosevelt’s?

Are there any surprises?
 
18. What other words in the corpus of Mount Rushmore presidents’ inaugural addresses do you want to analyze? Pick a word from rushmore_most_freq_words and find other words that are used similarly. Are you surprised by the words that are used in the same contexts?

19. Who are your favorite presidents? Do you want to see how their choice of words compares to presidents you view less favorably? Or controversial presidents?

Perhaps you want to analyze presidents by political party affiliation?

Create a new word embedding model trained on a corpus of sentences from the speeches of a selection of presidents that you decide. Find the words used similarly to “freedom”, and explore the embeddings of other words in the corpus.

How do the embeddings from your model compare to the other models you have built? What surprises do you find?