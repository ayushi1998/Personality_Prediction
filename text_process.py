import get_tweets

import csv
import re
import nltk
"""from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()"""

def cleaning(data):
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    """ data is a list  """
    big_data =[];
    label_big_data = {};
    for i in range(1,16):
        # Remove none alphabetic characters.

        data[0][i] = re.sub(r"http\S+", ' ', data[0][i])

        data[0][i] = data[0][i].replace("'","")

        data[0][i] = re.sub('[^A-Za-z]', ' ', data[0][i])

        data[0][i]= data[0][i].lower()
        big_data.append(data[0][i])




    """Tokenization and removing stop words from one sentence of the list of strings"""


    tokenized_data = word_tokenize(data[0][i])
    #nltk.download('stopwords')  #nltk.download('wordnet')


    for word in tokenized_data:
        if word in stopwords.words('english'):
            print(word)
            tokenized_data.remove(word)
    """Returning a list of tokenized 'clean' words"""
    return tokenized_data;



def lemmatization(tweet):

    from nltk.stem import WordNetLemmatizer
    from autocorrect import spell
    """Spell will autocorrect the sms language word to dictionary word"""
    wordnet_lemmatizer = WordNetLemmatizer()
    for j in range(len(tweet)):
        tweet[j] = wordnet_lemmatizer.lemmatize(spell(tweet[j]))

    tokenized_data=tweet;
    print(tokenized_data)
    return tokenized_data


def main():
    data=[]
    data = read_file();
    clean_tweet = cleaning(data)
    lemma_clean_tweet = lemmatization(clean_tweet)
    sentence_tweet = " ".join(lemma_clean_tweet)
    from sklearn.feature_extraction.text import CountVectorizer
    matrix = CountVectorizer(max_features=2)


if __name__ == "__main__":
    tweets = get_tweets.get_tweets_list('../project/mbti_1.csv')
    print(tweets)
    # main()
