import get_tweets
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

        for word in tokenized_data:
            if word in stopwords.words('english'):
                # print(word)
                tokenized_data.remove(word)

        all_tokenized.append(tokenized_data)

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
    for person in tweets:
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
    text_process(tweets)
    end = time.time()

    print(end-start)
