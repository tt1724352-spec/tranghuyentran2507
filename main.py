import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import requests
import re

# Download required NLTK data
nltk.download('vader_lexicon')

from nltk.sentiment import SentimentIntensityAnalyzer

# Download the dataset from Dropbox
url = "https://www.dropbox.com/scl/fi/w937d5gtkztw5ntztphb6/5000tweets.csv?rlkey=e5conogzw61vu2jht733pllai&st=7tdpcegm&dl=1"
response = requests.get(url)
with open('5000tweets.csv', 'wb') as file:
     file.write(response.content)

# Load the dataset
tweets = pd.read_csv('5000tweets.csv')

# Preprocess the data
tweets['text'] = tweets['text'].str.replace(r'http\S+|www.\S+', '', regex=True) # Removing URLs
tweets['text'] = tweets['text'].str.replace(r'@\S+', '', regex=True) # Removing mentions
tweets['text'] = tweets['text'].str.replace(r'#', '', regex=True) # Removing hashtag symbol
tweets['text'] = tweets['text'].str.lower() # Convert text to lowercase

# Analyze the sentiment
sia = SentimentIntensityAnalyzer()
tweets['polarity_score'] = tweets['text'].apply(lambda x: sia.polarity_scores(x)['compound'])
tweets['sentiment'] = np.where(tweets['polarity_score'] > 0.05, 'positive', np.where(tweets['polarity_score'] < -0.05, 'negative', 'neutral'))

# Visualize sentiment distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=tweets, x='sentiment', order=['positive', 'neutral', 'negative'])
plt.title('Sentiment Distribution of Tweets')
plt.show()

# Visualize sentiment trends over time (assuming there's a 'timestamp' column in datetime format)
if 'timestamp' in tweets.columns:
     # Pandas can choke on non-standard timezone abbreviations such as "PDT".
     # We'll try a forgiving conversion that coerces invalid values and
     # falls back to a simpler parse if needed.
     try:
         tweets['timestamp'] = pd.to_datetime(
             tweets['timestamp'],
             utc=True,            # convert everything to UTC
             errors='raise'
         )
     except ValueError:
         # remove unknown timezone abbreviations before parsing
         def _clean(ts: str) -> str:
             # strip out all-caps tokens of 2–4 letters (e.g. PDT, EST)
             return re.sub(r"\b[A-Z]{2,4}\b", "", ts).strip()

         tweets['timestamp'] = tweets['timestamp'].astype(str).apply(
             lambda x: pd.to_datetime(_clean(x), errors='coerce')
         )

     # if the column is still not datetime, coerce silently
     tweets['timestamp'] = pd.to_datetime(tweets['timestamp'], errors='coerce')

     tweets.set_index('timestamp', inplace=True)
     daily_sentiments = tweets.resample('D').apply(lambda x: x['sentiment'].value_counts(normalize=True) * 100)
     daily_sentiments = daily_sentiments.fillna(0).reset_index()

# Plotting
plt.figure(figsize=(12, 6))   
sns.lineplot(data=daily_sentiments, x='timestamp', y='positive', label='Positive')
sns.lineplot(data=daily_sentiments, x='timestamp', y='neutral', label='Neutral')
sns.lineplot(data=daily_sentiments, x='timestamp', y='negative', label  ='Negative')
plt.title('Daily Sentiment Trends of Tweets')   
plt.xlabel('Date')  
plt.ylabel('Percentage of Tweets')   
plt.legend()   
plt.show()
