import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# Load the model
model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5')

# 1. Contextual Preprocessing
def preprocess(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    return ' '.join([w for w in tokens if w not in stop_words])

# 2. Hierarchical Embedding
def get_hierarchical_embedding(content, category):
    content_emb = model.encode(preprocess(content))
    category_emb = model.encode(category)
    return np.concatenate([content_emb, category_emb * 0.5])  # Reduced weight for category

# 3. Weighted Similarity Scoring
def weighted_similarity(query_emb, doc_emb, query, doc, tfidf):
    cosine_sim = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
    
    # TF-IDF similarity
    query_tfidf = tfidf.transform([query]).toarray()
    doc_tfidf = tfidf.transform([doc]).toarray()
    tfidf_sim = np.dot(query_tfidf, doc_tfidf.T)[0][0]
    
    # Combine similarities
    return 0.7 * cosine_sim + 0.3 * tfidf_sim

# 4. Query Expansion
def expand_query(query):
    loan_types = ['home loan', 'personal loan', 'education loan', 'car loan', 'mortgage']
    expanded = query
    for lt in loan_types:
        if lt in query.lower():
            expanded += f" {lt} terms {lt} conditions {lt} details"
            break
    return expanded

# 5. Post-processing Re-ranking
def rerank(results, query):
    loan_types = ['home loan', 'personal loan', 'education loan', 'car loan', 'mortgage']
    query_type = next((lt for lt in loan_types if lt in query.lower()), None)
    
    if query_type:
        return sorted(results, key=lambda x: (query_type in x[1].lower(), x[0]), reverse=True)
    return results

# Main search function
def advanced_search(query, documents, categories):
    expanded_query = expand_query(query)
    query_emb = get_hierarchical_embedding(expanded_query, "loan")
    
    # Prepare TF-IDF vectorizer
    tfidf = TfidfVectorizer()
    tfidf.fit(documents + [expanded_query])
    
    # Calculate similarities
    similarities = [
        weighted_similarity(query_emb, get_hierarchical_embedding(doc, cat), 
                            expanded_query, doc, tfidf)
        for doc, cat in zip(documents, categories)
    ]
    
    # Initial ranking
    results = sorted(zip(similarities, documents), reverse=True)
    
    # Re-ranking
    return rerank(results, query)

# Example usage
documents = [
    "Our home loans offer competitive interest rates with flexible repayment options.",
    "Get a personal loan with quick approval and minimal documentation.",
    "Education loans to support your academic journey with easy repayment terms.",
    "Car loans with attractive interest rates and customizable tenure options."
]

categories = ["home loan", "personal loan", "education loan", "car loan"]

# Test queries
queries = [
    "What is the tenure for personal loan?",
    "Tell me about home loan interest rates",
    "How to apply for an education loan?"
]

for query in queries:
    print(f"\nQuery: {query}")
    results = advanced_search(query, documents, categories)
    for score, content in results:
        print(f"Score: {score:.4f}")
        print(f"Content: {content}")
