from save_data import load_data
from save_data import save_data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
import numpy as np
from bag_of_words import get_bag_of_words
from words_per_comment_feature import get_per_comment_features
from statistics import mean 

'''
#plot1 
df = pd.read_csv('../Project/mbti_1.csv')
df['words_per_comment'] = df['posts'].apply(lambda x: len(x.split())/50)
plt.figure(figsize=(5,5))
sns.stripplot(x='type', y='words_per_comment', data=df, size=4, jitter=True)
plt.ylabel('Words per comment')
plt.show()
'''
'''
#PLot 2
plt.figure(figsize=(5,5))
sns.set(style="darkgrid") 
print(df['type'])
sns.countplot(x="type", data=df, order = df['type'].value_counts().index)
plt.xlabel('Type')
plt.ylabel('Frequency')
plt.show()
'''

#plot 3
X_sentiment_features = load_data('sentiment_scores.dat')
X_bow,y = get_bag_of_words('data.dat')
X_wpc = load_data('words_per_comment.data')

X_avg_senti = [mean(l) for l in X_sentiment_features]
X_avg_senti = np.reshape(X_avg_senti, ((-1,1)))
X_bow = X_bow[:,:20]
X_wpc = np.reshape(X_wpc, ((8674,-1)))

X = np.append(X_bow, X_avg_senti , axis=1)
X = np.append(X, X_wpc, axis = 1)
#print(y)

print(X_bow)
save_data(X,'featur2_data.dat')

model = XGBClassifier()
model.fit(X, y)
plot_importance(model)
pyplot.show()