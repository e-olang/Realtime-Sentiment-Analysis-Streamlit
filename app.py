import tweepy as tw
import streamlit as st
import pandas as pd
import tensorflow as tf
from transformers import pipeline

import warnings
warnings.filterwarnings('ignore')

print('Tensorflow version ' + str(tf.__version__))


consumer_key = 'tYUR3KYI4wDbYzg8ehAiC7oDr'
consumer_secret = 'e0rQkDv3PcRko7VuogC6YPzJ9QSoHcgAWHI10GW2sGGzFQFOS4'
access_token = '993737466-ZHo3q228EZ7nWHBpmm0ZN3QJN7zIU5sH0ptBZRJx'
access_token_secret = 'ZyEjNLdOOJMKwtJm0lFzPOS1Y5SuZuUw95U8RImrtkSoo'


auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

classifier = pipeline('sentiment-analysis')

st.title('Live Twitter Sentiment Analysis with Tweepy and HuggingFace Transformers')
st.markdown('This app uses tweepy to get tweets from twitter based on the input name/phrase. It then processes the tweets through HuggingFace transformers pipeline function for sentiment analysis. The resulting sentiments and corresponding tweets are then put in a dataframe for display which is what you see as result.')

def run():
    with st.form(key='Enter name'):
        search_words = st.text_input('Enter the name for which you want to know the sentiment')
        number_of_tweets = st.number_input('Enter the number of latest tweets for which you want to know the sentiment(Maximum 50 tweets)', 0,50,10)
        submit_button = st.form_submit_button(label='Submit')
    if submit_button:
        tweets = tw.Cursor(api.search_tweets,q=search_words,lang="en").items(number_of_tweets)
        tweet_list = [i.text for i in tweets]
        p = [i for i in classifier(tweet_list)]
        q=[p[i]['label'] for i in range(len(p))]
        df = pd.DataFrame(list(zip(tweet_list, q)), columns = ['Latest '+str(number_of_tweets)+' Tweets'+' on '+search_words, 'sentiment'])
        st.write(df)

if __name__ == '__main__':
    run()