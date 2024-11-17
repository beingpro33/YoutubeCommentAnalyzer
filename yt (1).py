#!/usr/bin/env python
# coding: utf-8

# In[55]:


from googleapiclient.discovery import build
import pandas as pd

import pandas as pd
import re
import emoji
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from googletrans import Translator
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import nltk

# import seaborn as sns



# In[56]:


# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')



# In[57]:


# Set seed for consistent language detection results
DetectorFactory.seed = 0


# In[58]:


api_key = 'AIzaSyC9JU79CqHoZeWinCI6qe2xzrg64jbeGro'
channel_id = 'UCM5lnK0WJyzr6JEwkoT5VUw'

youtube = build('youtube', 'v3', developerKey=api_key)


# ## Function to get channel statistics

# In[59]:


def get_channel_stats(youtube, channel_id):
  
  request = youtube.channels().list(
    part = 'snippet, contentDetails, statistics',
    id = channel_id
  )

  response = request.execute()

  if 'items' not in response or len(response['items']) == 0:
        return {'error': 'Channel not found or invalid channel ID'}


  data = dict(Channel_name = response['items'][0]['snippet']['title'],
              Subscribers = response['items'][0]['statistics']['subscriberCount'],
              Views = response['items'][0]['statistics']['viewCount'],
              Total_videos = response['items'][0]['statistics']['videoCount'])
  
  return data


# In[60]:


get_channel_stats(youtube, channel_id)






#---------------
#Text with author name and serial number
#---------------

def fetch_comments(youtube, video_id):
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100
    )

    serial_number = 1

    while request:
        response = request.execute()
        
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append([serial_number, comment['textDisplay'], comment['authorDisplayName']])
            serial_number += 1
        
        # Check if there's a next page of comments
        request = youtube.commentThreads().list_next(request, response)
    
    return comments






# In[62]:


def print_comments(comments):
    
    for comment in comments:
        print(comment)
        print("-" * 40)  # Separator between comments


# In[63]:


def save_comments_to_csv(comments, csv_filename):
    # Create a DataFrame from the comments list
    df = pd.DataFrame(comments, columns=['Serial Number', 'Comment', 'Author'])
    
    # Save the DataFrame to a CSV file
    df.to_csv(csv_filename, index=False)
    print(f"Comments have been saved to {csv_filename}")


# In[64]:


video_id = 'TSZbuIqzi3w'
csv_filename = 'youtube_comments.csv'

# Fetch comments
comments = fetch_comments(youtube, video_id)

# Print comments
print_comments(comments)

# Save comments to CSV
save_comments_to_csv(comments, csv_filename)













# In[70]:


df = pd.read_csv('youtube_comments.csv')


# In[71]:


df.head()


# In[72]:


# Cell 2: Import necessary libraries for text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download stopwords and wordnet data
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# In[73]:


# Define emoji sentiment mapping
emoji_sentiment = {
    '😀': 1,  # positive
    '😊': 1, 
    '😃': 1,  # positive
    '😁': 1,  # positive
    '❤': 2,  # strong positive
    '😍': 2,  # strong positive
    '😢': -1, # negative
    '😠': -2, # strong negative
    '😩': -1, # negative
    '😐': 0,  # neutral
    '😕': 0   # neutral
    
    
}


# In[74]:


import re

def extract_emojis(text):
    # Updated emoji regex to match individual consecutive emojis
    emoji_pattern = re.compile(
        "[" 
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002700-\U000027BF"  # dingbats
        u"\U00002600-\U000026FF"  # miscellaneous symbols
        u"\U0001F900-\U0001F9FF"  # supplemental symbols & pictographs
        u"\U0001FA70-\U0001FAFF"  # symbols and pictographs extended
        u"\U00002500-\U00002BEF"  # other symbols
        "]", flags=re.UNICODE  # Remove the '+' so each emoji is captured individually
    )
    
    # Find all emojis in the text, capturing each emoji individually
    return emoji_pattern.findall(text)


# In[75]:


#  Function to calculate the sentiment score of emojis
def calculate_emoji_sentiment(emojis):
    
    # Check if there are any emojis, if not, return sentiment as 0
    if not emojis:
        return 0
    
    # Calculate the sentiment score based on the dictionary
    sentiment_score = sum(emoji_sentiment.get(emoji, 0) for emoji in emojis)
    
    return sentiment_score


# In[76]:


# Function to analyze emojis and print their sentiment score
def analyze_emoji(comment):
    emojis = extract_emojis(comment)

    # Calculate emoji sentiment
    emoji_sentiment_score = calculate_emoji_sentiment(emojis)

    return emoji_sentiment_score


# In[77]:


# Apply the emoji extraction and sentiment calculation
df['extracted_emojis'] = df['Comment'].apply(extract_emojis)
df['emoji_sentiment'] = df['extracted_emojis'].apply(calculate_emoji_sentiment)


# In[78]:


# Print the original comments and their extracted emojis
print(df[['Comment', 'extracted_emojis']].head(10))


# In[79]:


print(df[['Comment', 'emoji_sentiment']].head(10))


# In[80]:


df


# In[81]:


# Cell 3: Define preprocessing functions

# Function to Clean Comments
def clean_comment(comment):
    comment = re.sub(r'http\S+', '', comment)  # Remove URLs
    comment = re.sub(r'[^A-Za-z0-9\s]', '', comment)  # Remove special characters
    comment = comment.strip()  # Remove leading and trailing whitespace
    comment = comment.lower()  # Convert to lowercase
    return comment

# Function to remove stopwords
def remove_stopwords(tokens):
    return [word for word in tokens if word.lower() not in stop_words]

# Function to lemmatize tokens
def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]




# Tokenize, remove stopwords, and lemmatize each comment
def preprocess_comment(comment):
    cleaned_cmt = clean_comment(comment)
    tokens = word_tokenize(cleaned_cmt)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    # Extract emojis
    # emojis = extract_emojis(comment)
    

    # Calculate emoji sentiment
    # emoji_sentiment_score = calculate_emoji_sentiment(emojis)

    return ' '.join(tokens)


# In[82]:


# Apply preprocessing to the 'Comment' column and store both tokens and emoji sentiment
df['processed_comment'] = df['Comment'].apply(preprocess_comment)

# Show the processed comments and their corresponding emoji sentiment scores
df[['Comment', 'processed_comment']].head(20)


# In[83]:


# Cell 5: Print the output for a few comments
for i in range(5):  # Print the first 5 comments
    print(f"Original: {df['Comment'][i]}")
    print(f"Processed: {df['processed_comment'][i]}")
    print("-" * 50)


# In[84]:


# Show the processed comments and emoji sentiment
df[['Comment', 'processed_comment', 'extracted_emojis', 'emoji_sentiment']].head(20)


# In[85]:


# # Apply preprocessing to the 'Comment' column and store both tokens and emoji sentiment
# df[['processed_comment', 'emoji_sentiment']] = df['Comment'].apply(lambda x: preprocess_comment(x)).apply(pd.Series)

# # Show the processed comments and their corresponding emoji sentiment scores
# df[['Comment', 'processed_comment', 'emoji_sentiment']].head(20)


# In[86]:


# Cell 6: Detect language of each comment
from langdetect import detect, LangDetectException

# Function to detect language
def detect_language(comment):
    try:
        return detect(comment)
    except LangDetectException:
        return 'unknown'

# Apply language detection
df['language'] = df['Comment'].apply(detect_language)

# Show the language detection results
df[['Comment', 'language']].head()


# In[87]:


# Cell 7: Print language detection output for first 5 comments
for i in range(5):
    print(f"Comment: {df['Comment'][i]}")
    print(f"Detected Language: {df['language'][i]}")
    print("-" * 50)


# In[88]:


# Cell 8: Translate non-English comments
from googletrans import Translator

translator = Translator()

# Function to translate comments to English
def translate_comment(comment, target_lang='en'):
    try:
        translated = translator.translate(comment, dest=target_lang)
        return translated.text
    except Exception as e:
        return comment  # Return the original comment if translation fails

# Translate only non-English comments
df['translated_comment'] = df.apply(lambda row: translate_comment(row['processed_comment']) if row['language'] != 'en' else row['processed_comment'], axis=1)

# Show translated comments
df[['processed_comment', 'translated_comment']].head()


# In[89]:


# Cell 9: Print the output of translated comments for first 5 non-English comments
non_english_comments = df[df['language'] != 'en'].head(10)
for i, row in non_english_comments.iterrows():
    print(f"Original: {row['Comment']}")
    print(f"Translated: {row['translated_comment']}")
    print("-" * 50)


# In[90]:


df


# In[91]:


# Function to classify the sentiment based on VADER's compound score
def classify_sentiment(comment):
    vader_score = analyzer.polarity_scores(comment)['compound']
    if vader_score >= 0.05:
        return 'positive', vader_score
    elif vader_score <= -0.05:
        return 'negative', vader_score
    else:
        return 'neutral', vader_score


# In[92]:


import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to classify the sentiment based on VADER's compound score
def classify_sentiment(vader_score):
    if vader_score >= 0.05:
        return 'positive'
    elif vader_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Apply VADER to calculate sentiment scores for translated comments
df['vader_score'] = df['translated_comment'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

# Classify the sentiment based on the VADER score
df['sentiment_class'] = df['vader_score'].apply(classify_sentiment)

# Save the updated DataFrame to a new CSV file
df.to_csv('classified_comments.csv', index=False)

# Display the first few rows of the DataFrame
print(df.head())


# In[93]:


df




# Define weights for combining VADER and Emoji sentiment scores
vader_weight = 0.7
emoji_weight = 0.3

# Calculate combined sentiment score
df['combined_score'] = (df['vader_score'] * vader_weight) + (df['emoji_sentiment'] * emoji_weight)

# Map combined score to sentiment class
def combined_to_label(score):
    if score > 0.05:
        return 'positive'
    elif score < -0.05:
        return 'negative'
    else:
        return 'neutral'

df['combined_sentiment'] = df['combined_score'].apply(combined_to_label)

df


# In[106]:


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

# Load pre-trained BERT model and tokenizer manually
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Function to predict sentiment with confidence thresholds for "neutral"
def get_bert_sentiment_batch(comments, neutral_threshold=0.7):
    # Tokenize the input comments in batch
    inputs = tokenizer(comments, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get logits and apply softmax to get probabilities
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    
    sentiments = []
    for prob in probabilities:
        pos_prob = prob[1].item()  # Probability for positive
        neg_prob = prob[0].item()  # Probability for negative
        
        # Set sentiment based on probability thresholds
        if max(pos_prob, neg_prob) < neutral_threshold:
            sentiment = 'neutral'
        else:
            sentiment = 'positive' if pos_prob > neg_prob else 'negative'
        
        sentiments.append({
            'sentiment': sentiment,
            'positive_score': pos_prob,
            'negative_score': neg_prob
        })
    
    return sentiments

# Apply BERT model to comments in batches
batch_size = 5  # Set batch size for processing
comments = df['translated_comment'].tolist()
results = []

for i in range(0, len(comments), batch_size):
    batch_comments = comments[i:i+batch_size]
    batch_results = get_bert_sentiment_batch(batch_comments)
    results.extend(batch_results)

# Append results to DataFrame
df['bert_sentiment'] = [result['sentiment'] for result in results]
df['bert_positive_score'] = [result['positive_score'] for result in results]
df['bert_negative_score'] = [result['negative_score'] for result in results]

# Display DataFrame with enhanced output
print(df[['translated_comment', 'bert_sentiment', 'bert_positive_score', 'bert_negative_score']])


# In[107]:


import matplotlib.pyplot as plt

# Visualization of comparison
labels = df['Serial Number']
x = range(len(df))

# Set up a larger figure for clarity
plt.figure(figsize=(14, 7))

# Plot Combined sentiment score (VADER + Emoji)
bar_width = 0.35
combined_bar = plt.bar(x, df['combined_score'], color='purple', alpha=0.6, width=bar_width, label="Combined Sentiment (VADER + Emoji)")

# Plot BERT sentiment (as scores are not available, this is a label plot)
bert_values = [1 if sentiment == "positive" else -1 if sentiment == "negative" else 0 for sentiment in df['bert_sentiment']]
bert_bar = plt.bar([i + bar_width for i in x], bert_values, color='green', alpha=0.6, width=bar_width, label="BERT Sentiment (Positive=1, Negative=-1, Neutral=0)")

# Adding labels and title
plt.xticks([i + bar_width / 2 for i in x], labels)
plt.xlabel('Serial Number')
plt.ylabel('Sentiment Score / Class')
plt.title('Comparison of Combined Sentiment (VADER + Emoji) vs. BERT Sentiment')

# Annotate bars with values
for i in range(len(df)):
    plt.text(i, df['combined_score'][i] + 0.1, round(df['combined_score'][i], 2), ha='center', color='black')
    plt.text(i + bar_width, bert_values[i] + 0.1, df['bert_sentiment'][i], ha='center', color='black')

# Legend
plt.legend()

plt.tight_layout()
plt.show()


# In[100]:


import matplotlib.pyplot as plt
import numpy as np

# Create a larger figure
plt.figure(figsize=(14, 7))

# Define the positions for the bars (side by side)
x = np.arange(len(df))  # The x-axis positions for each comment
bar_width = 0.35  # Bar width

# Plot Combined Sentiment Scores (VADER + Emoji)
combined_bar = plt.bar(x - bar_width/2, df['combined_score'], color='purple', alpha=0.7, width=bar_width, label="Combined Sentiment (VADER + Emoji)")

# Plot BERT Sentiment (convert to numerical values for visualization)
bert_values = [1 if sentiment == "positive" else -1 if sentiment == "negative" else 0 for sentiment in df['bert_sentiment']]
bert_bar = plt.bar(x + bar_width/2, bert_values, color='green', alpha=0.7, width=bar_width, label="BERT Sentiment (Positive=1, Negative=-1, Neutral=0)")

# Customize the ticks and labels
plt.xticks(x, df['Serial Number'], rotation=0)
plt.xlabel('Serial Number')
plt.ylabel('Sentiment Score / Class')
plt.title('Comparison of Combined Sentiment (VADER + Emoji) vs. BERT Sentiment')

# Add value annotations for clarity
for i in range(len(df)):
    plt.text(x[i] - bar_width/2, df['combined_score'][i] + 0.05, round(df['combined_score'][i], 2), ha='center', va='bottom', color='black', fontsize=10)
    plt.text(x[i] + bar_width/2, bert_values[i] + 0.05, df['bert_sentiment'][i], ha='center', va='bottom', color='black', fontsize=10)

# Adding a legend
plt.legend()

# Improve layout and show plot
plt.tight_layout()
plt.show()


# In[108]:


import matplotlib.pyplot as plt

# Calculate sentiment distribution for Combined Model (VADER + Emoji)
combined_sentiment_counts = df['combined_sentiment'].value_counts()
combined_labels = combined_sentiment_counts.index
combined_sizes = combined_sentiment_counts.values

# Calculate sentiment distribution for BERT Model
bert_sentiment_counts = df['bert_sentiment'].value_counts()
bert_labels = bert_sentiment_counts.index
bert_sizes = bert_sentiment_counts.values

# Create a figure with subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 7))

# Pie chart for Combined Sentiment (VADER + Emoji)
axs[0].pie(combined_sizes, labels=combined_labels, autopct='%1.1f%%', startangle=90, colors=['purple', 'orange', 'grey'])
axs[0].set_title("Sentiment Distribution - Combined Model (VADER + Emoji)")

# Pie chart for BERT Sentiment
axs[1].pie(bert_sizes, labels=bert_labels, autopct='%1.1f%%', startangle=90, colors=['green', 'lightblue', 'lightgrey'])
axs[1].set_title("Sentiment Distribution - BERT Model")

# Display the pie charts
plt.tight_layout()
plt.show()

