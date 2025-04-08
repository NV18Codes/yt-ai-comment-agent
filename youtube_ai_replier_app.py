import streamlit as st

# ✅ Must be the first Streamlit command
st.set_page_config(page_title="YouTube AI Comment Replier", layout="centered")

# 🔁 Imports
from pytube import YouTube
import pandas as pd
from youtube_comment_downloader import YoutubeCommentDownloader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
    set_seed,
)
import plotly.express as px


# ✅ Load BERT-like sentiment model (more compatible)
@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
    )


# ✅ Load GPT-2 for AI replies
@st.cache_resource
def load_gpt2_model():
    generator = pipeline("text-generation", model="distilgpt2")
    set_seed(42)
    return generator


# Initialize models
sentiment_pipeline = load_sentiment_model()
gpt2_generator = load_gpt2_model()


# ✅ Scrape YouTube comments (top comments only)
def get_comments(url, limit=10):
    downloader = YoutubeCommentDownloader()
    comments = []
    for comment in downloader.get_comments_from_url(url, sort_by="top"):
        comments.append(comment["text"])
        if len(comments) >= limit:
            break
    return comments


# ✅ Analyze & reply to comments
def analyze_comments(comments):
    data = []
    for comment in comments:
        sentiment_result = sentiment_pipeline(comment)[0]
        label = sentiment_result["label"]
        reply = (
            gpt2_generator(
                f"User: {comment}\nAI:", max_length=60, num_return_sequences=1
            )[0]["generated_text"]
            .split("AI:")[-1]
            .strip()
        )
        data.append({"Comment": comment, "Sentiment": label, "AI Reply": reply})
    return pd.DataFrame(data)


# ✅ Streamlit App UI
st.title("🤖 YouTube AI Comment Replier (No API Needed)")
st.markdown(
    "Paste a YouTube video URL, and this app will fetch top comments, analyze their sentiment, and auto-reply using AI!"
)

url = st.text_input("📥 Enter YouTube Video URL")
num_comments = st.slider("🔢 Number of Comments", 5, 30, 10)

if url and st.button("🔍 Analyze"):
    with st.spinner("⏳ Analyzing comments..."):
        try:
            yt = YouTube(url)
            title = yt.title
            description = yt.description

            comments = get_comments(url, limit=num_comments)
            if not comments:
                st.warning("No comments found.")
            else:
                df = analyze_comments(comments)

                st.subheader("🎬 Video Info")
                st.markdown(f"**Title:** {title}")
                st.markdown(f"**Description:** {description[:300]}...")

                st.subheader("💬 Comments, Sentiment & AI Replies")
                st.dataframe(df)

                # 📊 Sentiment Chart
                sentiment_counts = df["Sentiment"].value_counts().reset_index()
                sentiment_counts.columns = ["Sentiment", "Count"]
                fig = px.pie(
                    sentiment_counts,
                    names="Sentiment",
                    values="Count",
                    title="📊 Sentiment Breakdown",
                )
                st.plotly_chart(fig)

                # 📥 Download CSV
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📁 Download Results as CSV",
                    csv,
                    "youtube_ai_analysis.csv",
                    "text/csv",
                )

        except Exception as e:
            st.error(f"❌ Error: {e}")
