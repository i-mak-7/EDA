#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score


# In[3]:


cd C:\Users\UCER PC\desktop\mayank\tweet-sentiment-extraction/


# In[4]:


train_tweets = pd.read_csv('train_tweets.csv')
test_tweets = pd.read_csv('test_tweets.csv')


# In[5]:


train_tweets = train_tweets[['label','tweet']]
train_tweets


# In[6]:


test = test_tweets[['tweet']]
test


# In[7]:


train_tweets['length'] = train_tweets['tweet'].apply(len)


# In[8]:


train_tweets


# In[19]:


sns.barplot('label','length',data = train_tweets)
plt.title('Average Word Length vs label')


# In[20]:


sns.countplot(x= 'label',data = train_tweets)


# In[9]:


def text_processing(tweet):
    
    #Generating the list of words in the tweet (hastags and other punctuations removed)
    def form_sentence(tweet):
        tweet_blob = TextBlob(tweet)
        return ' '.join(tweet_blob.words)
    new_tweet = form_sentence(tweet)
    
     
    #Removing stopwords and words with unusual symbols
    def no_user_alpha(tweet):
        tweet_list = [ele for ele in tweet.split() if ele != 'user']
        clean_tokens = [t for t in tweet_list if re.match(r'[^\W\d]*$', t)]
        clean_s = ' '.join(clean_tokens)
        clean_mess = [word for word in clean_s.split() if word.lower() not in stopwords.words('english')]
        return clean_mess
    no_punc_tweet = no_user_alpha(new_tweet)
    
    #Normalizing the words in tweets 
    def normalization(tweet_list):
        lem = WordNetLemmatizer()
        normalized_tweet = []
        for word in tweet_list:
            normalized_text = lem.lemmatize(word,'v')
            normalized_tweet.append(normalized_text)
        return str(normalized_tweet)
    
    
    return normalization(no_punc_tweet)
    


# In[10]:


train_tweets['tweet_list'] = train_tweets['tweet'].apply(text_processing)
test_tweets['tweet_list'] = test_tweets['tweet'].apply(text_processing)


# In[13]:


print(type(train_tweets['tweet_list'] [0]))


# In[13]:


def converttostr(input_seq, seperator):
   # Join all the strings in list
   final_str = seperator.join(input_seq)
   return final_str


# In[14]:


seperator = ' '
converttostr(train_tweets['tweet_list'][0], seperator)


# In[86]:



for i in range(len(train_tweets['tweet_list']) ):
    tweet=converttostr(train_tweets['tweet_list'][i], seperator)
    print(tweet)
    


# In[87]:


for i in range(len(train_tweets['tweet_list']) ):
    train_tweets['tweet_list'][i] = train_tweets['tweet_list'][i] 

print( "The twwet " + str(train_tweets['tweet_list'][i]) + "." )


# In[ ]:


str_tweet= ''.join(tweet)
    print(str_tweet)


# In[17]:


train_tweets['tweet_list_str']


# In[25]:


train_tweets[train_tweets['label']==1].drop('tweet',axis=1).head()


# In[14]:


train_tweets


# In[21]:


X = train_tweets['tweet']
y = train_tweets['label']
test = test_tweets['tweet']

print(type(train_tweets['tweet'][0]))


# In[96]:





# In[16]:



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_tweets['tweet_list'], train_tweets['label'], test_size=0.2)


# In[17]:


#Machine Learning Pipeline
pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_processing)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])
pipeline.fit(X_train,y_train)


# In[22]:


predictions = pipeline.predict(test)



# In[ ]:


print(classification_report(predictions,y_test))
print ('\n')
print(confusion_matrix(predictions,y_test))
print(accuracy_score(predictions,y_test)) 


# In[25]:


s = pd.Series(predictions)


# In[40]:





# In[41]:


import csv

with open('submission.csv', 'w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(["ItemID", "Sentiment"])
    s.to_csv("submission.csv", index=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




