import get_tweets
import save_data

import time
import csv
import re
import nltk
import os

import numpy as np

from nltk.tag import pos_tag
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
        if not without_stop_words:
            continue
        all_tokenized.append(without_stop_words)

    return all_tokenized



def lemmatization(list_of_tokens):

    """Spell will autocorrect the sms language word to dictionary word"""
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized = []
    tag_dict = {'ADJ': 'a', 'ADJ_SAT':'s', 'ADV':'r', 'NOUN':'n', 'VERB':'v','DET':'n','NUM':'n','PRON':'n','X':'n','.':'n','CONJ':'n','ADP':'s','PRT':'n'}
    with_pos_tags = pos_tag(list_of_tokens,tagset='universal')
    for j in range(len(with_pos_tags)):
        lemmatized_word = wordnet_lemmatizer.lemmatize(with_pos_tags[j][0],tag_dict[with_pos_tags[j][1]])
        lemmatized.append(lemmatized_word)

    return lemmatized


def text_process(tweets):
    final = []
    temp_file_number = 1
    for person in tweets:
        filename = "../data/%d.dat" % temp_file_number
        if os.path.exists(filename):
            continue
        all_tokenized = cleaning(person[1])
        all_lematized = []
        for list_of_tokens in all_tokenized:
            lemmatized = lemmatization(list_of_tokens)
            all_lematized.append(lemmatized)
        to_append = [person[0],all_lematized]
        final.append(to_append)
        save_data.save_data(to_append, filename)
        temp_file_number+=1

    return final

if __name__ == "__main__":
    start = time.time()
    tweets = get_tweets.get_tweets_list('../project/mbti_1.csv')
    size = len(tweets)
    # print(tweets)

    data = text_process(tweets)
    save_data.save_data(data,"data.dat")
    end = time.time()
    print(end - start)
    # print(data)

    # print(save_data.load_data("data.dat"))
