
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('../Project/mbti_1.csv')
df['words_per_comment'] = df['posts'].apply(lambda x: len(x.split())/50)
plt.figure(figsize=(5,5))
sns.stripplot(x='type', y='words_per_comment', data=df, size=4, jitter=True)
plt.ylabel('Words per comment')
plt.show()