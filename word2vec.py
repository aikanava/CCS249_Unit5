import wikipedia
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# --- Step 1: Get Wikipedia summaries ---
topics = ["Computer science", "Artificial intelligence", "Deep learning", "Python", "Computer programming"]
documents = []

for topic in topics:
    try:
        documents.append(wikipedia.summary(topic, sentences=2))
    except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError) as e:
        documents.append(f"Error: Could not retrieve Wikipedia page for '{topic}'")

# --- Step 2: Tokenize and preprocess ---
stop_words = set(stopwords.words('english'))

def preprocess(doc):
    # Tokenize, remove punctuation, lowercase, and remove stopwords
    tokens = word_tokenize(doc.lower())
    return [word for word in tokens if word not in stop_words and word not in string.punctuation]

tokenized_docs = [preprocess(doc) for doc in documents]

# --- Step 3: Train Word2Vec model ---
model = Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, min_count=1, workers=4, sg=1, epochs=10)

# --- Step 4: Convert documents to average word vectors ---
def get_doc_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

doc_vectors = np.array([get_doc_vector(doc, model) for doc in tokenized_docs])

# --- Step 5: Prepare labels and train classifier ---
labels = list(range(len(documents)))  # Label: 0 to 4 for each topic

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(doc_vectors, labels, test_size=0.2, random_state=42)

classifier = LogisticRegression(max_iter=1000, solver='liblinear')  # Using a more suitable solver

classifier.fit(X_train, y_train)

# --- Step 6: Predict and evaluate ---
predictions = classifier.predict(X_test)

print("Logistic Regression Classification Report:\n")
print(classification_report(y_test, predictions, zero_division=1))
