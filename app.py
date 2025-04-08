import streamlit as st
import pandas as pd
from googleapiclient.discovery import build
from transformers import pipeline, set_seed, AutoTokenizer, AutoModelForCausalLM
from textblob import TextBlob

# Load GPT-2 model using CPU
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
set_seed(42)


# YouTube API setup
def get_youtube_comments(api_key, video_id, max_results=10):
    youtube = build("youtube", "v3", developerKey=api_key)
    comments = []
    response = (
        youtube.commentThreads()
        .list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            textFormat="plainText",
        )
        .execute()
    )

    for item in response["items"]:
        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(comment)
    return comments


# Generate AI reply using GPT-2
def generate_reply(comment):
    prompt = f"User comment: {comment}\nAI reply:"
    response = generator(prompt, max_length=60, num_return_sequences=1)
    reply = response[0]["generated_text"].split("AI reply:")[-1].strip()
    return reply


# Sentiment analysis using TextBlob
def detect_sentiment(comment):
    polarity = TextBlob(comment).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"


# Streamlit UI
st.title("🎥 YouTube Comment AI Replier")

api_key = st.text_input("🔑 Enter YouTube Data API Key", type="password")
video_id = st.text_input("📹 Enter YouTube Video ID", value="dQw4w9WgXcQ")
max_comments = st.slider("🔢 Number of Comments to Fetch", 1, 50, 10)

if st.button("Generate AI Replies"):
    with st.spinner("Fetching comments and generating replies..."):
        try:
            comments = get_youtube_comments(api_key, video_id, max_comments)
            data = []
            for comment in comments:
                reply = generate_reply(comment)
                sentiment = detect_sentiment(comment)
                data.append(
                    {"Comment": comment, "Sentiment": sentiment, "AI Reply": reply}
                )

            df = pd.DataFrame(data)
            st.success("✅ Done generating replies!")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download CSV", csv, "youtube_ai_replies.csv", "text/csv"
            )

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
