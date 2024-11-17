import streamlit as st
from googleapiclient.discovery import build
import pandas as pd
import re
import emoji
from langdetect import detect, DetectorFactory
from googletrans import Translator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from wordcloud import WordCloud
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from io import BytesIO
import requests
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import nltk
from langdetect import detect, LangDetectException
from sklearn.metrics import accuracy_score, classification_report
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer

# Download stopwords and wordnet data
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Initialize necessary libraries
DetectorFactory.seed = 0
api_key = 'AIzaSyC9JU79CqHoZeWinCI6qe2xzrg64jbeGro'  # Replace with your actual API key
youtube = build('youtube', 'v3', developerKey=api_key)

# Streamlit UI
st.title("YouTube Comment Sentiment Analysis")

# Functions for processing
def get_youtube_video_id(url):
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return None

def get_channel_id(video_id):
    request = youtube.videos().list(part="snippet", id=video_id)
    response = request.execute()
    if "items" not in response or len(response["items"]) == 0:
        return None
    return response["items"][0]["snippet"]["channelId"]

def get_channel_details(channel_id):
    """Fetch channel details using YouTube Data API."""
    request = youtube.channels().list(part="snippet", id=channel_id)
    response = request.execute()
    if "items" not in response or not response["items"]:
        return None, None

    channel = response["items"][0]
    name = channel["snippet"]["title"]
    image_url = channel["snippet"]["thumbnails"]["high"]["url"]
    return name, image_url

def get_video_details(video_id):
    request = youtube.videos().list(part="snippet,statistics", id=video_id)
    response = request.execute()
    if "items" not in response or len(response["items"]) == 0:
        return None
    video_data = response["items"][0]
    details = {
        "title": video_data["snippet"]["title"],
        "description": video_data["snippet"]["description"],
        "published_at": video_data["snippet"]["publishedAt"],
        "view_count": video_data["statistics"]["viewCount"],
        "like_count": video_data["statistics"].get("likeCount", "N/A"),
        "comment_count": video_data["statistics"].get("commentCount", "N/A"),
    }
    return details

def get_comments(video_id, max_comments):
    comments = []
    request = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=min(max_comments, 100))
    while request and len(comments) < max_comments:
        response = request.execute()
        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]
            comments.append([comment["textDisplay"], comment["authorDisplayName"]])
            if len(comments) >= max_comments:
                break
        request = youtube.commentThreads().list_next(request, response)
    return comments

# def clean_comment(comment):
#     comment = re.sub(r'http\S+', '', comment)
#     comment = re.sub(r'[^A-Za-z0-9\s]', '', comment)
#     return comment.strip().lower()

# def extract_emojis(text):
#     return [c for c in text if c in emoji.EMOJI_DATA]

# def calculate_emoji_sentiment(emojis):
#     emoji_sentiment = {
#         '😀': 1, '😊': 1, '😃': 1, '😁': 1, '❤': 2, '😍': 2, '😢': -1, '😠': -2, '😩': -1, '😐': 0, '😕': 0
#     }
#     return sum([emoji_sentiment.get(e, 0) for e in emojis])

# def detect_language(comment):
#     try:
#         return detect(comment)
#     except Exception:
#         return 'unknown'

# def translate_comment(comment, target_lang='en'):
#     translator = Translator()
#     try:
#         return translator.translate(comment, dest=target_lang).text
#     except Exception:
#         return comment

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

#  Function to calculate the sentiment score of emojis
def calculate_emoji_sentiment(emojis):
    
    # Check if there are any emojis, if not, return sentiment as 0
    if not emojis:
        return 0
    
    # Calculate the sentiment score based on the dictionary
    sentiment_score = sum(emoji_sentiment.get(emoji, 0) for emoji in emojis)
    
    return sentiment_score
    
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

# Function to detect language
def detect_language(comment):
    try:
        return detect(comment)
    except LangDetectException:
        return 'unknown'


translator = Translator()

# Function to translate comments to English
def translate_comment(comment, target_lang='en'):
    try:
        translated = translator.translate(comment, dest=target_lang)
        return translated.text
    except Exception as e:
        return comment  # Return the original comment if translation fails


# Sentiment Analysis
analyzer = SentimentIntensityAnalyzer()

def classify_vader_sentiment(comment):
    score = analyzer.polarity_scores(comment)['compound']
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

def get_bert_sentiment(comment):
    inputs = tokenizer(comment, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    if probabilities[0][1].item() > 0.6:
        return 'positive'
    elif probabilities[0][0].item() > 0.6:
        return 'negative'
    else:
        return 'neutral'
    

def evaluate_models(df):
    # Evaluation against manually labeled data
    true_labels = df['True_Label']  # Assuming the DataFrame contains manually labeled data
    vader_predictions = df['vader_sentiment']
    bert_predictions = df['bert_sentiment']

    # Calculate accuracies
    vader_accuracy = accuracy_score(true_labels, vader_predictions)
    bert_accuracy = accuracy_score(true_labels, bert_predictions)

    # Classification reports
    vader_report = classification_report(true_labels, vader_predictions)
    bert_report = classification_report(true_labels, bert_predictions)

    return vader_accuracy, bert_accuracy, vader_report, bert_report

def analyze_video(video_url):
    video_id = get_youtube_video_id(video_url)
    if not video_id:
        st.error("Invalid YouTube video URL.")
        return

    channel_id = get_channel_id(video_id)
    channel_name, channel_image_url = get_channel_details(channel_id)
    if not channel_name or not channel_image_url:
        st.error("Failed to fetch channel details.")
        return

    st.write(f"**video id:** {video_id}")
    st.write(f"**channel id:** {channel_id}")
    st.write(f"**Channel Name:** {channel_name}")
    response = requests.get(channel_image_url)
    img = Image.open(BytesIO(response.content))
    st.image(img, caption=channel_name, use_column_width=True)

    video_details = get_video_details(video_id)
    if video_details:
        st.subheader("Video Details")
        st.write(video_details)

    max_comments = st.slider("Number of comments to analyze", 10, 500, 100)
    comments = get_comments(video_id, max_comments)
    df = pd.DataFrame(comments, columns=["Comment", "Author"])

    # Load internal CSV file for True Labels
    true_labels_path = "labeled_comments.csv"
    try:
        true_labels_df = pd.read_csv(true_labels_path)
        if "True_Label" not in true_labels_df.columns:
            st.error("The internal CSV file must contain a 'True_Label' column.")
            return
        df['True_Label'] = true_labels_df['True_Label']
    except FileNotFoundError:
        st.error(f"Internal CSV file not found at {true_labels_path}. Please ensure the file exists.")
        return

    df['processed_comment'] = df['Comment'].apply(preprocess_comment)
    df['language'] = df['Comment'].apply(detect_language)
    # Translate only non-English comments
    df['translated_comment'] = df.apply(lambda row: translate_comment(row['processed_comment']) if row['language'] != 'en' else row['processed_comment'], axis=1)
    df['extracted_emojis'] = df['Comment'].apply(extract_emojis)
    df['emoji_sentiment'] = df['extracted_emojis'].apply(calculate_emoji_sentiment)
    # df['cleaned_comment'] = df['Comment'].apply(clean_comment)
    # df['emojis'] = df['Comment'].apply(extract_emojis)
    # df['emoji_sentiment'] = df['emojis'].apply(calculate_emoji_sentiment)
    # df['language'] = df['Comment'].apply(detect_language)
    # df['translated_comment'] = df['cleaned_comment'].apply(
    #     lambda x: translate_comment(x) if detect_language(x) != 'en' else x
    # )
    df['vader_sentiment'] = df['translated_comment'].apply(classify_vader_sentiment)
    df['bert_sentiment'] = df['translated_comment'].apply(get_bert_sentiment)

    # Evaluate models
    vader_accuracy, bert_accuracy, vader_report, bert_report = evaluate_models(df)

    # Display processed data
    st.write("Processed Data Sample:")
    st.dataframe(df.head())

    # Sentiment Distributions
    vader_counts = df['vader_sentiment'].value_counts()
    bert_counts = df['bert_sentiment'].value_counts()

    # Bar Charts
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(x=vader_counts.index, y=vader_counts.values, ax=axes[0], palette="Set2")
    sns.barplot(x=bert_counts.index, y=bert_counts.values, ax=axes[1], palette="Set3")
    axes[0].set_title("VADER Sentiment Distribution")
    axes[1].set_title("BERT Sentiment Distribution")
    st.pyplot(fig)

    # Pie Charts
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].pie(vader_counts, labels=vader_counts.index, autopct='%1.1f%%', colors=sns.color_palette("Set2"))
    axes[0].set_title("VADER Sentiment Pie Chart")
    axes[1].pie(bert_counts, labels=bert_counts.index, autopct='%1.1f%%', colors=sns.color_palette("Set3"))
    axes[1].set_title("BERT Sentiment Pie Chart")
    st.pyplot(fig)

     # Display evaluation results
    st.subheader("Model Evaluation")
    st.write(f"**VADER Accuracy:** {vader_accuracy:.2f}")
    st.write(f"**BERT Accuracy:** {bert_accuracy:.2f}")
    st.write("**VADER Classification Report**")
    st.json(vader_report)
    st.write("**BERT Classification Report**")
    st.json(bert_report)

    # Visualize evaluation results
    labels = ['VADER', 'BERT']
    accuracies = [vader_accuracy, bert_accuracy]

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=labels, y=accuracies, palette="coolwarm", ax=ax)
    ax.set_title("Model Accuracy Comparison")
    ax.set_ylabel("Accuracy")
    st.pyplot(fig)

    # Word Cloud
    st.subheader("Word Cloud for Comments")
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(df['processed_comment']))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

    # CSV Download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name='youtube_comments_analysis.csv',
        mime='text/csv',
    )

video_url = st.text_input("Enter YouTube Video URL", "")
if video_url:
    analyze_video(video_url)