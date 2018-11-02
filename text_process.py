import get_tweets
import save_data
import time

import csv
import re
import nltk


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from autocorrect import spell

def cleaning(list_of_tweets):
    """ data is a list  """
    all_tokenized = []
    for string in list_of_tweets:
        without_links = re.sub(r"http\S+", ' ', string)
        without_punct = re.sub('[^A-Za-z]', ' ', without_links)
        lower_case = without_punct.lower()

        """Tokenization and removing stop words from one sentence
        of the list of strings"""
        tokenized_data = word_tokenize(lower_case)
        without_stop_words = []
        for word in tokenized_data:
            if word not in stopwords.words('english'):
                # print(word)
                without_stop_words.append(word)

        all_tokenized.append(without_stop_words)

    return all_tokenized



def lemmatization(list_of_tokens):

    """Spell will autocorrect the sms language word to dictionary word"""
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized = []

    for j in range(len(list_of_tokens)):
        lemmatized.append(wordnet_lemmatizer.lemmatize(list_of_tokens[j]))

    return lemmatized


def text_process(tweets):
    final = []
    i = 0
    for person in tweets:
        if i>1:
            break
        i+=1
        all_tokenized = cleaning(person[1])
        all_lematized = []
        for list_of_tokens in all_tokenized:
            lemmatized = lemmatization(list_of_tokens)
            all_lematized.append(lemmatized)
        final.append([person[0],all_lematized])
    return final

if __name__ == "__main__":
    tweets = get_tweets.get_tweets_list('../project/mbti_1.csv')
    # print(tweets)
    start = time.time()
    save_data.save_data(text_process(tweets),"data.dat")
    end = time.time()

    # print(save_data.load_data("data.dat"))

    print(end-start)
