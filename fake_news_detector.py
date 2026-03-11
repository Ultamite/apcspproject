import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
import nltk
import matplotlib.pyplot as plt

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

st.set_page_config(page_title="Fake News Detector", layout="wide")

@st.cache_resource
def train_model(num_samples):
    # 1. Load Data
    true = pd.read_csv("News_dataset/True.csv", encoding='latin-1').fillna("")
    false = pd.read_csv("News_dataset/Fake.csv", encoding='latin-1').fillna("")
    true["true_or_false"] = "True"
    false["true_or_false"] = "False"

    # 2. Combine and Sample
    data = pd.concat([true, false])
    actual_samples = min(num_samples, len(data))
    data = data.sample(n=actual_samples, random_state=42).reset_index(drop=True)

    # 3. Preprocessing
    stop_words = set(stopwords.words('english'))
    def preprocess(text_input):
        tokens = word_tokenize(text_input)
        filtered_list = [word.lower() for word in tokens if word.lower() not in stop_words and word.isalpha()]
        return " ".join(filtered_list)

    data["content"] = (data["title"] + " " + data["text"]).apply(preprocess)

    # 4. Prepare Features and Labels (THE FIX)
    X = data['content'].to_numpy()
    y = data["true_or_false"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Build Pipeline and Grid Search
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    param_grid = {
        'tfidf__max_features': [1000, 2000],
        'clf__C': [0.1, 1, 10, 100],
        'clf__penalty': ['l2'],
        'clf__solver': ['liblinear'] 
    }

    gs = GridSearchCV(estimator=pipeline, 
                      param_grid=param_grid, 
                      cv=5, 
                      scoring='accuracy',
                      n_jobs=-1) 

    gs.fit(X_train, y_train)
    return gs, X_test, y_test

## --- Streamlit UI ---
st.title("📰 Fake News Classifier")
st.markdown("This app uses a Logistic Regression model with Grid Search to identify potentially false news articles.")

# Sidebar for configuration
with st.sidebar:
    st.header("Settings")
    num_samples = st.number_input("Enter number of samples:", min_value=10, max_value=40000, value=1000)
    train_btn = st.button("Train Model")

if train_btn or 'model' in st.session_state:
    if train_btn:
        with st.spinner("Training model... this may take a moment."):
            model, x_test, y_test = train_model(num_samples)
            st.session_state['model'] = model
            st.session_state['x_test'] = x_test
            st.session_state['y_test'] = y_test
        st.success("Model trained successfully!")

    # Layout for Results
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Model Performance")
        best_model = st.session_state['model']
        st.write(f"**Best Params:** {best_model.best_params_}")
        st.write(f"**Best CV Accuracy:** {best_model.best_score_:.4f}")
        
    with col2:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(best_model, st.session_state['x_test'], st.session_state['y_test'], ax=ax)
        st.pyplot(fig)

    st.divider()

    # Prediction Section
    st.subheader("Detect Fake News")
    user_input = st.text_area("Paste news content here:", placeholder="The moon is made of green cheese...")

    if st.button("Analyze News"):
        if user_input.strip():
            prediction = best_model.predict([user_input])[0]
            
            if prediction == "True":
                st.success("✅ This news appears to be **TRUE**.")
            else:
                st.error("🚨 This news appears to be **FALSE**.")
        else:
            st.warning("Please enter some text first.")
else:
    st.info("Adjust the sample size in the sidebar and click 'Train Model' to begin.")

