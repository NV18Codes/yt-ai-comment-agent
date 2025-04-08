import streamlit as st
from yt_dlp import YoutubeDL
from youtube_comment_downloader import YoutubeCommentDownloader
from transformers import pipeline
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="YouTube Comment AI Replier", layout="centered")

st.markdown("## 🤖 YouTube Comment AI Replier")

url = st.text_input("📽️ Enter YouTube Video URL or Video ID", "")
num_comments = st.slider("💬 Number of Comments", 5, 50, 10)

if st.button("✨ Generate AI Replies") and url:
    try:
        with st.spinner("Fetching video info..."):
            ydl_opts = {"quiet": True}
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                video_title = info.get("title", "N/A")
                video_description = info.get("description", "N/A")

        st.success(f"🎥 Video Title: {video_title}")
        st.info(f"📝 Description:\n{video_description[:300]}...")

        with st.spinner("🔍 Fetching comments..."):
            downloader = YoutubeCommentDownloader()
            comments_gen = downloader.get_comments_from_url(url, sort_by=0)
            comments = []
            for i, comment in enumerate(comments_gen):
                if i >= num_comments:
                    break
                comments.append(comment["text"])

        if comments:
            st.success("✅ Comments fetched. Generating replies...")

            reply_model = pipeline("text-generation", model="distilgpt2")
            replies = []
            sentiments = []

            for comment in comments:
                try:
                    sentiment_score = float(TextBlob(comment).sentiment.polarity)
                    label = (
                        "Positive"
                        if sentiment_score > 0
                        else "Negative" if sentiment_score < 0 else "Neutral"
                    )
                except Exception as e:
                    sentiment_score = 0
                    label = "Neutral"

                reply = reply_model(
                    f"Reply to: {comment}\n", max_length=50, num_return_sequences=1
                )[0]["generated_text"]
                replies.append(reply.strip())
                sentiments.append(label)

            df = pd.DataFrame(
                {"Comment": comments, "Sentiment": sentiments, "AI Reply": replies}
            )

            st.dataframe(df)

            # Sentiment Pie Chart
            sentiment_counts = df["Sentiment"].value_counts()
            fig, ax = plt.subplots()
            ax.pie(
                sentiment_counts,
                labels=sentiment_counts.index,
                autopct="%1.1f%%",
                startangle=90,
            )
            ax.axis("equal")
            st.pyplot(fig)

            # CSV Download
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📁 Download CSV", csv, "comments_ai_replies.csv", "text/csv"
            )

        else:
            st.warning("⚠️ No comments found.")

    except Exception as e:
        st.error(f"❌ Exception: {str(e)}")
