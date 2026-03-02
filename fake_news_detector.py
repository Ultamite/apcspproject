import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
import nltk

# 1. Load Data
true = pd.read_csv("News_dataset/True.csv", encoding='latin-1').fillna("")
false = pd.read_csv("News_dataset/Fake.csv", encoding='latin-1').fillna("")

true["true_or_false"] = "True"
false["true_or_false"] = "False"

# 2. Combine and Sample
data = pd.concat([true, false])

data = data.sample(n=5000, random_state=42).reset_index(drop=True)

# 3. Preprocessing
data["content"] = data["title"] + " " + data["text"]
stop_words = set(stopwords.words('english'))

def preprocess(text_input):
    # Traversal and Loop
    tokens = word_tokenize(text_input)
    filtered_list = []
    
    for word in tokens:
        if word.lower() not in stop_words and word.isalpha():
            filtered_list.append(word.lower())
            
    return " ".join(filtered_list)

data["content"] = data["content"].apply(preprocess)

# 4. Prepare Features and Labels
X = data['content']
y = data["true_or_false"].values.ravel()

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

GS = GridSearchCV(estimator=pipeline, 
                  param_grid=param_grid, 
                  cv=5, 
                  scoring='accuracy',
                  n_jobs=-1) 

# 6. Fit and Evaluate
print("Starting Grid Search...")
GS.fit(X_train, y_train)

train_accuracy = GS.score(X_train, y_train)
test_accuracy = GS.score(X_test, y_test)

print("\n--- Results ---")
print("Best Parameters:", GS.best_params_)
print("Best Cross-Validation Accuracy:", GS.best_score_)
print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

y_pred = GS.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. Visualize
ConfusionMatrixDisplay.from_estimator(GS, X_test, y_test)

while True:
    message = input("Enter news to see if it is fake:")
    if not message.strip():
        break

    pred = GS.predict([message])[0]

    if pred == "False":
        print("This news appears to be False")
    elif pred == "True":

        print("This news appears to be True")
