#!/usr/bin/env python
# coding: utf-8

# In[8]:


import tweepy
from requests_oauthlib import OAuth1Session

client = tweepy.Client(bearer_token='bearer token from dev portal')

query = 'ouragan -is:retweet lang:fr'
tweets = client.search_recent_tweets(query=query, tweet_fields=['context_annotations', 'created_at'], max_results=100)

with open("dataset2.txt", "a") as file:
    for tweet in tweets.data:
        print(tweet.text)
        print("\n---------------------------\n")
        print(tweet.text,file=file)
        print("\n---------------------------\n",file=file)
        if len(tweet.context_annotations) > 0:
            #print(tweet.context_annotations)
            print(tweet.context_annotations,file=file)


# In[3]:


#!pip install requests_oauthlib


# In[ ]:



