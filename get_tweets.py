import csv
import linecache


def get_tweets_list(filename):
    total_lines = 0
    with open(filename,'r') as f:
        for line in f:
            total_lines += 1

    # print(total_lines)

    all_tweets = []

    for i in range(2,total_lines):
        clean = []
        line = linecache.getline(filename, i)
        label = line[:4]
        line = line[5:]
        line = line.replace('\"','')
        line = line.replace('\'','')
        line = line.replace(",",'')
        tweets = list(filter(None,line.strip('\n').split("|||")))
        clean.append(label)
        clean.append(tweets)
        all_tweets.append(clean)
    return all_tweets


if __name__ == '__main__':
    all_tweets = get_tweets_list('./mbti_1.csv')
    print(all_tweets)
