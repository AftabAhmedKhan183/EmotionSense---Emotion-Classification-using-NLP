Emotion Classification using NLP

An NLP-based machine learning project that analyzes text and predicts the emotion expressed in it. The project demonstrates the complete text classification pipeline, from preprocessing and feature extraction to model training and prediction through an interactive application.

🚀 Features

- Text preprocessing and cleaning
- Count Vectorization
- TF-IDF Vectorization
- Multinomial Naive Bayes classification
- Logistic Regression classification
- Emotion prediction from user-provided text
- Interactive user interface
- Complete ML workflow implemented in Python

🧠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLP
- TF-IDF
- Count Vectorizer
- Multinomial Naive Bayes
- Logistic Regression

📂 Project Structure

Emotion-Classification-NLP/
│
├── app/
│   └── app.py
│
├── notebook/
│   └── emotion_classification.ipynb
│
├── data/
│   └── train.txt
│
├── README.md
└── requirements.txt

🔄 Workflow

Raw Text
   ↓
Text Preprocessing
   ↓
Feature Extraction
   ↓
Count Vectorizer / TF-IDF
   ↓
Model Training
   ↓
Multinomial Naive Bayes / Logistic Regression
   ↓
Emotion Prediction
   ↓
Interactive Application

📊 Models Used

Multinomial Naive Bayes

Multinomial Naive Bayes is used as a baseline text classification model and works particularly well with discrete text features such as word counts and TF-IDF representations.

Logistic Regression

Logistic Regression is also trained for emotion classification and provides a strong and interpretable baseline for text-based classification problems.

🔤 Feature Extraction

Two common NLP feature extraction techniques are explored:

Count Vectorizer

Converts text into a matrix of token counts, representing how frequently words occur in each document.

TF-IDF

TF-IDF assigns importance to words based on how frequently they occur in a document while reducing the importance of words that appear frequently across the entire dataset.

🎯 Prediction

The trained model takes a text input from the user and predicts the emotion associated with that text.

Example:

Input:
"I am extremely happy with my results!"

Prediction:
Joy

📓 Jupyter Notebook

The "emotion_classification.ipynb" notebook contains the experimentation and machine learning pipeline, including:

- Data exploration
- Text preprocessing
- Feature extraction
- Model training
- Model evaluation
- Comparison of classification approaches

▶️ How to Run

1. Clone the repository

git clone https://github.com/your-username/Emotion-Classification-NLP.git
cd Emotion-Classification-NLP

2. Install dependencies

pip install -r requirements.txt

3. Run the application

Use the command required by the framework used in the project.

For example, if the application uses Streamlit:

streamlit run app/app.py

📌 Future Improvements

- Add more emotion categories
- Experiment with advanced NLP techniques
- Try deep learning models such as LSTM and Transformers
- Improve model accuracy through hyperparameter tuning
- Deploy the application online

👨‍💻 Author

Aftab Ahmed Khan

CSE Student | NIT Patna

---

⭐ If you find this project useful, consider giving the repository a star!
