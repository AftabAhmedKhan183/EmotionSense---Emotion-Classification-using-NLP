import streamlit as st
import joblib
import re
import string
import nltk
import pandas as pd

# PAGE CONFIG

st.set_page_config(
    page_title="EmotionSense AI",
    page_icon="🧠",
    layout="wide"
)

# LOAD MODEL FILES

@st.cache_resource
def load_models():

    model = joblib.load("emotion_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    emotion_map = joblib.load("emotion_map.pkl")

    return model, vectorizer, emotion_map


try:
    model, vectorizer, emotion_map = load_models()

except Exception as e:
    st.error("❌ Could not load the model files.")
    st.code(str(e))
    st.stop()


# LOAD STOPWORDS

@st.cache_resource
def load_stopwords():

    try:
        from nltk.corpus import stopwords
        try:
            return set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            return set(stopwords.words("english"))

    except Exception:
        return set()


STOP_WORDS = load_stopwords()


# EMOTION INFORMATION

emotion_data = {
    "Sadness": {
        "emoji": "😔",
        "description": "A feeling of sadness, disappointment or loss."
    },

    "Anger": {
        "emoji": "😡",
        "description": "A feeling of frustration, irritation or anger."
    },

    "Love": {
        "emoji": "❤️",
        "description": "A feeling of affection, care or attachment."
    },

    "Surprise": {
        "emoji": "😮",
        "description": "A feeling caused by something unexpected."
    },

    "Fear": {
        "emoji": "😨",
        "description": "A feeling of worry, nervousness or fear."
    },

    "Joy": {
        "emoji": "😄",
        "description": "A feeling of happiness, excitement or pleasure."
    }
}

# TITLE

st.title("🧠 EmotionSense AI")

st.caption(
    "NLP-powered Emotion Classification using Machine Learning"
)

st.divider()


# INTRODUCTION

st.subheader("Understand the emotion behind words")

st.write(
    "EmotionSense analyzes your text using Natural Language "
    "Processing and Machine Learning and predicts the emotion "
    "expressed in it."
)

st.info(
    "Model: Count Vectorizer + Logistic Regression | "
    "Test Accuracy: 89.13%"
)


# =========================================================
# INPUT SECTION
# =========================================================

st.subheader("📝 Analyze Your Text")

text = st.text_area(
    "Enter your text",
    height=180,
    placeholder=(
        "Example: I finally achieved my dream and "
        "I'm extremely happy!"
    )
)

st.caption(f"{len(text)} characters")


# =========================================================
# EXAMPLES
# =========================================================

st.write("**Try an example:**")

col1, col2, col3, col4 = st.columns(4)

example1 = "I finally got the job I've always wanted!"
example2 = "I can't believe they did this to me."
example3 = "I feel completely alone and hopeless."
example4 = "That surprise completely shocked me!"


with col1:

    if st.button(
        "😄 Joy Example",
        use_container_width=True
    ):

        text = example1


with col2:

    if st.button(
        "😡 Anger Example",
        use_container_width=True
    ):

        text = example2


with col3:

    if st.button(
        "😔 Sadness Example",
        use_container_width=True
    ):

        text = example3


with col4:

    if st.button(
        "😮 Surprise Example",
        use_container_width=True
    ):

        text = example4


st.divider()

# PREPROCESSING

def clean_text(text):

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # Remove numbers
    text = re.sub(
        r"\d+",
        "",
        text
    )

    # Remove emojis / non-ASCII characters
    text = text.encode(
        "ascii",
        "ignore"
    ).decode(
        "ascii"
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Remove stopwords
    words = []

    for word in text.split():
        if word not in STOP_WORDS:
            words.append(word)

    return " ".join(words)


# EMOTION MAPPING

def get_emotion_label(mapping, prediction):
    prediction_int = int(prediction)
    # Mapping:
    # {0: "joy", 1: "sadness", ...}

    if prediction_int in mapping:
        value = mapping[prediction_int]
        if isinstance(value, str):
            return value

    # Mapping:
    # {"joy": 0, "sadness": 1, ...}

    for key, value in mapping.items():
        try:
            if int(value) == prediction_int:
                return str(key)

        except:
            continue

    return str(prediction)


# =========================================================
# ANALYZE BUTTON
# =========================================================

analyze = st.button(
    "✨ Analyze Emotion",
    type="primary",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if analyze:
    if not text.strip():
        st.warning(
            "Please enter some text first."
        )

    else:
        with st.spinner(
            "Analyzing your text..."
        ):

            # Preprocess
            processed_text = clean_text(text)

            # Vectorization
            vectorized_text = vectorizer.transform(
                [processed_text]
            )

            # Prediction
            prediction = model.predict(
                vectorized_text
            )[0]

            # Probabilities
            probabilities = model.predict_proba(
                vectorized_text
            )[0]

            # Emotion
            emotion = get_emotion_label(
                emotion_map,
                prediction
            )

            emotion = emotion.capitalize()

            # Confidence
            confidence = max(probabilities) * 100


        # =================================================
        # RESULT
        # =================================================

        data = emotion_data.get(
            emotion,
            {
                "emoji": "🧠",
                "description": "Emotion detected."
            }
        )

        st.divider()

        st.subheader("🎯 Prediction")

        result_col1, result_col2 = st.columns(
            [1, 2]
        )

        with result_col1:

            st.metric(
                label="Detected Emotion",
                value=f"{data['emoji']} {emotion}"
            )

        with result_col2:

            st.metric(
                label="Model Confidence",
                value=f"{confidence:.2f}%"
            )


        st.success(
            f"{data['emoji']} The model predicts **{emotion}**."
        )

        st.caption(
            data["description"]
        )


        # PROBABILITY DISTRIBUTION

        st.subheader(
            "📊 Emotion Probability Distribution"
        )

        probability_data = []

        for class_id, probability in zip(
            model.classes_,
            probabilities
        ):

            label = get_emotion_label(
                emotion_map,
                class_id
            )

            probability_data.append(
                {
                    "Emotion": label.capitalize(),
                    "Probability": probability * 100
                }
            )


        probability_data.sort(
            key=lambda x: x["Probability"],
            reverse=True
        )


        for item in probability_data:
            label = item["Emotion"]
            probability = item["Probability"]
            emoji = emotion_data.get(
                label,
                {"emoji": "🧠"}
            )["emoji"]
            st.write(
                f"{emoji} **{label} — {probability:.2f}%**"
            )

            st.progress(
                min(
                    int(probability),
                    100
                )
            )

        # RESULT TABLE

        st.subheader(
            "📋 Prediction Details"
        )

        result_df = pd.DataFrame(
            probability_data
        )

        result_df["Probability"] = (
            result_df["Probability"]
            .round(2)
        )

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# MODEL INFORMATION
# =========================================================

st.divider()

st.subheader("🤖 About the Model")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Training Samples",
        "16,000"
    )


with col2:

    st.metric(
        "Emotion Classes",
        "6"
    )


with col3:

    st.metric(
        "Test Accuracy",
        "89.13%"
    )


with col4:

    st.metric(
        "Algorithm",
        "Logistic Regression"
    )


# =========================================================
# ML PIPELINE
# =========================================================

st.subheader("⚙️ Machine Learning Pipeline")

st.write(
    "The application follows this pipeline:"
)

pipeline = st.columns(5)

with pipeline[0]:
    st.info("📝\n\nInput Text")

with pipeline[1]:
    st.info("🧹\n\nPreprocessing")

with pipeline[2]:
    st.info("🔢\n\nCount Vectorizer")

with pipeline[3]:
    st.info("🤖\n\nLogistic Regression")

with pipeline[4]:
    st.success("🎯\n\nEmotion")


# =========================================================
# MODEL COMPARISON
# =========================================================

st.subheader("📈 Model Comparison")

comparison = pd.DataFrame(
    {
        "Vectorization": [
            "TF-IDF",
            "TF-IDF",
            "Count Vectorizer",
            "Count Vectorizer"
        ],

        "Algorithm": [
            "Multinomial Naive Bayes",
            "Logistic Regression",
            "Multinomial Naive Bayes",
            "Logistic Regression"
        ],

        "Accuracy": [
            "66.09%",
            "87.84%",
            "76.81%",
            "89.13%"
        ]
    }
)

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)


st.success(
    "🏆 Best Model: Count Vectorizer + Logistic Regression — 89.13%"
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "EmotionSense AI • Natural Language Processing • "
    "Machine Learning • Python • Scikit-learn • Streamlit"
)