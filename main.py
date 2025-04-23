from tf import compute_tf
from tf_idf import compute_idf, compute_tfidf
import wikipedia

topics = ["Computer science", "Artificial intelligence", "Deep learning", "Python", "Computer programming"]
documents = []

for topic in topics:
    try:
        summary = wikipedia.summary(topic, sentences=2)  # Use 2 sentences per topic
        documents.append(summary)
    except Exception as e:
        print(f"Error fetching summary for {topic}: {e}")
        documents.append("")

tokenized_docs = [doc.lower().split() for doc in documents]
vocabulary = sorted(set(word for doc in tokenized_docs for word in doc))

# Compute TF
tf_vectors = [compute_tf(doc, vocabulary) for doc in tokenized_docs]

# Compute IDF
idf = compute_idf(tokenized_docs, vocabulary)

# Compute TF-IDF
tfidf_vectors = [compute_tfidf(tf, idf, vocabulary) for tf in tf_vectors]

print("Documents Summary:")
for i, doc in enumerate(documents):
    print(f"\nDocument {i+1} ({topics[i]}):")
    print(doc)

# TF
print("\n--- Term Frequency (Raw Count-Based) ---")
for i, tf_vector in enumerate(tf_vectors):
    print(f"\nDocument {i + 1} ({topics[i]}):")
    for term in vocabulary:
        if tf_vector[term] > 0:  # Only display terms with non-zero frequency
            print(f"{term}: {tf_vector[term]}")

# IDF
print("\n--- Inverse Document Frequency (IDF) ---")
for term in vocabulary:
    print(f"{term}: {idf[term]}")

# TF-IDF (Top 5 Terms)
print("\n--- TF-IDF Vectors (Top 5 Terms) ---")
for i, tfidf_vector in enumerate(tfidf_vectors):
    print(f"\nDocument {i + 1} ({topics[i]}):")
    # Display top 5 TF-IDF terms (sorted by value)
    sorted_terms = sorted(tfidf_vector.items(), key=lambda item: item[1], reverse=True)[:5]
    for term, score in sorted_terms:
        print(f"{term}: {round(score, 4)}")
