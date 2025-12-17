import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"  # FastAPI endpoint

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Climate Change Sentiment Analysis",
    layout="centered"
)

# ---------------------------------
# Title & Description
# ---------------------------------
st.title("🌍 Climate Change Sentiment Analysis")
st.write(
    """
    This dashboard sends climate-related tweets to a FastAPI backend
    and displays sentiment, confidence, misinformation detection,
    and emotion analysis results.
    """
)

st.markdown("---")

# ---------------------------------
# User Input
# ---------------------------------
tweet_text = st.text_area(
    "Enter a climate-related tweet:",
    height=120,
    placeholder="Example: Climate change is real and we need immediate action."
)

# ---------------------------------
# Submit Button
# ---------------------------------
if st.button("Analyze Tweet"):
    if tweet_text.strip() == "":
        st.warning("⚠ Please enter some text before submitting.")
    else:
        with st.spinner("Contacting backend API..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"text": tweet_text},
                    timeout=10
                )

                # Handle non-200 responses
                if response.status_code != 200:
                    st.error("❌ Backend returned an error.")
                    st.write("Status Code:", response.status_code)
                    st.write(response.text)

                else:
                    result = response.json()

                    # ---------------------------------
                    # Display Results
                    # ---------------------------------
                    st.success("✅ Analysis Completed")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Sentiment")
                        st.write(result.get("sentiment", "N/A"))

                        st.subheader("Confidence")
                        st.write(round(result.get("confidence", 0.0), 2))

                    with col2:
                        st.subheader("Misinformation Label")
                        st.write(result.get("misinformation", "N/A"))

                        st.subheader("Emotion")
                        st.write(result.get("emotion", "N/A"))

            except requests.exceptions.Timeout:
                st.error("⏳ Request timed out. Backend may be slow or unavailable.")

            except requests.exceptions.ConnectionError:
                st.error("🔌 Unable to connect to backend API.")

            except Exception as e:
                st.error("❌ An unexpected error occurred.")
                st.write(e)

# ---------------------------------
# Footer
# ---------------------------------
st.markdown("---")
st.caption("Major Project | Climate Change Sentiment Analysis using ClimateBERT")
