import streamlit as st

# ✅ MUST BE FIRST Streamlit command
st.set_page_config(page_title="YouTube AI Comment Replier", layout="centered")

# Continue with the rest of your imports
from pytube import YouTube
import pandas as pd
from youtube_comment_downloader import YoutubeCommentDownloader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, set_seed
import torch
import plotly.express as px
import streamlit as st
from pytube import YouTube
import pandas as pd
from youtube_comment_downloader import YoutubeCommentDownloader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, set_seed
import torch
import plotly.express as px

# Load BERT model for sentiment
@st.cache_resource
def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Load GPT2 model for replies
@st.cache_resource
def load_gpt2_model():
    generator = pipeline("text-generation", model="distilgpt2")
    set_seed(42)
    return generator

sentiment_pipeline = load_sentiment_model()
gpt2_generator = load_gpt2_model()

# Scrape comments using youtube-comment-downloader
def get_comments(url, limit=10):
    downloader = YoutubeCommentDownloader()
    comments = []
    for comment in downloader.get_comments_from_url(url, sort_by="top"):
        comments.append(comment["text"])
        if len(comments) >= limit:
            break
    return comments

# Sentiment & reply generator
def analyze_comments(comments):
    data = []
    for comment in comments:
        sentiment_result = sentiment_pipeline(comment)[0]
        label = sentiment_result["label"].capitalize()
        reply = gpt2_generator(f"User: {comment}\nAI:", max_length=60, num_return_sequences=1)[0]["generated_text"].split("AI:")[-1].strip()
        data.append({"Comment": comment, "Sentiment": label, "AI Reply": reply})
    return pd.DataFrame(data)

# Streamlit UI
st.set_page_config(page_title="YouTube AI Comment Replier", layout="centered")
st.title("🎬 YouTube AI Comment Replier (No API Needed)")

url = st.text_input("📥 Paste a YouTube Video URL")

num_comments = st.slider("🔢 Number of Comments to Analyze", 5, 30, 10)

if url and st.button("Analyze Video"):
    with st.spinner("⏳ Fetching video and analyzing comments..."):
        try:
            yt = YouTube(url)
            title = yt.title
            description = yt.description
            comments = get_comments(url, limit=num_comments)
            df = analyze_comments(comments)

            st.subheader("🎥 Video Summary")
            st.markdown(f"**Title:** {title}")
            st.markdown(f"**Description:** {description[:300]}...")

            st.subheader("💬 Comments, Sentiment & AI Replies")
            st.dataframe(df)

            # Sentiment breakdown chart
            sentiment_counts = df["Sentiment"].value_counts().reset_index()
            sentiment_counts.columns = ["Sentiment", "Count"]
            fig = px.pie(sentiment_counts, names="Sentiment", values="Count", title="Sentiment Breakdown")
            st.plotly_chart(fig)

            # Download
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", csv, "youtube_ai_comments.csv", "text/csv")

        except Exception as e:
            st.error(f"Something went wrong: {e}")
