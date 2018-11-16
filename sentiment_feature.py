import get_tweets
from textblob import TextBlob

def feature_extraction(tweets):

    feature = []
    l = []
    #list_of_tweets 
    for person in tweets:
        l.append(person[0])
        list_of_tweets = person[1]
        polarity_value=[]
        for sentance in list_of_tweets:
            test = TextBlob(sentance)
            polarity_value.append(tes.sentiment.polarity)

        feature.append(polarity_value)

        

    return feature,l
            
            
            
        

if __name__ == '__main__':
    start = time.time()
    tweets = get_tweets.get_tweets_list('../project/mbti_1.csv')
    X , y = feature_extraction(tweets([0:5])
    end = time.time()
    print("Time",end - start)
    print(X)
    

        
        
        
    
