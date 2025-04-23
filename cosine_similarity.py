import math
import wikipedia
from collections import Counter

def get_documents_from_wikipedia(topics, sentences=2):
    """Fetches summaries from Wikipedia for a list of topics."""
    documents = []
    for topic in topics:
        try:
            documents.append(wikipedia.summary(topic, sentences=sentences))
        except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError):
            documents.append(f"Error fetching '{topic}'")
    return documents

def compute_tf(tokens, vocab):
    return {term: tokens.count(term) for term in vocab}

def compute_idf(docs, vocab):
    N = len(docs)
    return {term: math.log(N / (sum(term in doc for doc in docs) or 1)) for term in vocab}

def compute_tfidf(tf, idf, vocab):
    return {term: tf.get(term, 0) * idf.get(term, 0) for term in vocab}

def cosine_similarity(vec1, vec2, vocab):
    dot_product = sum(vec1[term] * vec2[term] for term in vocab)
    magnitude1 = math.sqrt(sum(val**2 for val in vec1.values()))
    magnitude2 = math.sqrt(sum(val**2 for val in vec2.values()))
    return dot_product / (magnitude1 * magnitude2) if magnitude1 and magnitude2 else 0.0

if __name__ == "__main__":
    topics = ["Computer science", "Artificial intelligence", "Deep learning", "Python", "Computer programming"]
    docs = get_documents_from_wikipedia(topics)

    # Tokenization and vocabulary
    tokenized_docs = [doc.lower().split() for doc in docs]
    vocab = sorted(set(word for doc in tokenized_docs for word in doc))

    # TF, IDF, and TF-IDF computations
    tf_vectors = [compute_tf(doc, vocab) for doc in tokenized_docs]
    idf = compute_idf(tokenized_docs, vocab)
    tfidf_vectors = [compute_tfidf(tf, idf, vocab) for tf in tf_vectors]

    # Output
    print("\n--- Document Summaries ---")
    for i, doc in enumerate(docs):
        print(f"\nDocument {i+1} ({topics[i]}): {doc}")

    print("\n--- TF-IDF Vectors (Top 5 Terms) ---")
    for i, tfidf_vector in enumerate(tfidf_vectors):
        top_terms = sorted(tfidf_vector.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\nDocument {i+1} ({topics[i]}):")
        for term, score in top_terms:
            print(f"{term}: {round(score, 4)}")

    print("\n--- Cosine Similarity Matrix ---")
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            similarity = cosine_similarity(tfidf_vectors[i], tfidf_vectors[j], vocab)
            print(f"Doc {i + 1} vs. Doc {j + 1}: {round(similarity, 4)}")
