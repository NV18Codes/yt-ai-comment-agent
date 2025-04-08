import streamlit as st
from yt_dlp import YoutubeDL
from youtube_comment_downloader import YoutubeCommentDownloader
from transformers import pipeline, set_seed
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="YouTube Comment AI Replier", layout="centered")

st.markdown("## 🤖 YouTube Comment AI Replier")

url = st.text_input("📽️ Enter YouTube Video URL or Video ID", "")
num_comments = int(st.slider("💬 Number of Comments", 5, 50, 10))

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
            comments_gen = downloader.get_comments_from_url(url)
            comments = []
            for i, comment in enumerate(comments_gen):
                if i >= num_comments:
                    break
                comments.append(comment["text"])

        if comments:
            st.success("✅ Comments fetched. Generating replies...")

            reply_model = pipeline("text-generation", model="distilgpt2")
            set_seed(42)
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
                except Exception:
                    sentiment_score = 0
                    label = "Neutral"

                prompt = f'Reply to this YouTube comment in a fun, lighthearted, and satisfying way: "{comment}"\n'
                reply = reply_model(prompt, max_new_tokens=60, num_return_sequences=1)[
                    0
                ]["generated_text"]
                clean_reply = reply.replace(prompt.strip(), "").strip().split("\n")[0]
                replies.append(clean_reply)
                sentiments.append(label)

            df = pd.DataFrame(
                {"Comment": comments, "Sentiment": sentiments, "AI Reply": replies}
            )

            st.dataframe(df)

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

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📁 Download CSV", csv, "comments_ai_replies.csv", "text/csv"
            )

        else:
            st.warning("⚠️ No comments found.")

    except Exception as e:
        st.error(f"❌ Exception: {str(e)}")
