import csv
import re
import nltk
f = open('tweets.csv',encoding="utf-8")
reader = csv.reader(f)
next(reader) # skip header
data = []
i=0;

for row in reader:
    i=i+1;
    data.append(row)
    if(i==1):
        break;
#List of strings

print(data[0][25])
for i in range(1,26):
    # Remove none alphabetic characters.
    data[0][i] = re.sub(r"http\S+", ' ', data[0][i])
    
    #print(data[0][i])
    data[0][i] = re.sub('[^A-Za-z]', ' ', data[0][i])
    

    data[0][i]= data[0][i].lower()
    


from nltk.tokenize import word_tokenize
tokenized_data = word_tokenize(data[0][i])

nltk.download('stopwords')
from nltk.corpus import stopwords 

for word in tokenized_data:
    if word in stopwords.words('english'):
        print(word)
        tokenized_data.remove(word)

print(tokenized_data)

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
"""
for j in range(len(tokenized_data)):
    tokenized_data[j] = stemmer.stem(tokenized_data[j])
"""

from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')
wordnet_lemmatizer = WordNetLemmatizer()
for j in range(len(tokenized_data)):
    tokenized_data[j] = wordnet_lemmatizer.lemmatize(tokenized_data[j])

print(tokenized_data)

from autocorrect import spell
for j in range(len(tokenized_data)):
    tokenized_data[j] = stemmer.stem(spell(tokenized_data[j]))

print(tokenized_data)
