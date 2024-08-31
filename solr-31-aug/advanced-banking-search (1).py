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

# Define banking terms globally
banking_terms = {
    'loan': ['loan', 'borrow', 'finance', 'credit'],
    'savings': ['savings', 'deposit', 'interest rate', 'yield'],
    'credit card': ['credit card', 'card', 'cashback', 'rewards'],
    'checking': ['checking', 'current account', 'overdraft'],
    'investment': ['investment', 'mutual fund', 'stocks', 'bonds'],
    'mortgage': ['mortgage', 'home loan', 'property finance'],
    'insurance': ['insurance', 'policy', 'coverage', 'premium'],
    'online banking': ['online banking', 'mobile banking', 'digital'],
    'customer service': ['customer service', 'support', 'helpline']
}

# 1. Contextual Preprocessing
def preprocess(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    return ' '.join([w for w in tokens if w not in stop_words])

# 2. Hierarchical Embedding
def get_hierarchical_embedding(content, category):
    content_emb = model.encode(preprocess(content))
    category_emb = model.encode(category)
    return np.concatenate([content_emb, category_emb * 0.3])  # Reduced weight for category

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
    expanded = query
    for category, terms in banking_terms.items():
        if any(term in query.lower() for term in terms):
            expanded += f" {category} {' '.join(terms)}"
    
    return expanded

# 5. Post-processing Re-ranking
def rerank(results, query):
    banking_categories = list(banking_terms.keys())
    query_category = next((cat for cat in banking_categories if any(term in query.lower() for term in banking_terms[cat])), None)
    
    if query_category:
        return sorted(results, key=lambda x: (query_category in x[1].lower(), x[0]), reverse=True)
    return results

# Main search function
def advanced_banking_search(query, documents, categories):
    expanded_query = expand_query(query)
    query_emb = get_hierarchical_embedding(expanded_query, "banking")
    
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
    "Our savings accounts offer competitive interest rates with flexible deposit options.",
    "Get a credit card with cashback rewards and travel benefits.",
    "Home mortgages with attractive interest rates and customizable tenure options.",
    "Open a checking account with no minimum balance and free online bill pay.",
    "Invest in our range of mutual funds managed by expert financial advisors.",
    "24/7 customer support available through our mobile banking app.",
    "Comprehensive insurance policies to protect what matters most to you.",
    "Business loans to help your company grow with flexible repayment terms."
]

categories = ["savings", "credit card", "mortgage", "checking", "investment", "online banking", "insurance", "loan"]

# Test queries
queries = [
    "What are the interest rates for savings accounts?",
    "How do I apply for a credit card with travel rewards?",
    "Tell me about your mortgage options for first-time homebuyers",
    "What are the fees associated with your checking accounts?",
    "I need information about your investment products",
    "How can I contact customer support outside of business hours?",
    "What types of insurance do you offer?",
    "What are the requirements for a small business loan?"
]

for query in queries:
    print(f"\nQuery: {query}")
    results = advanced_banking_search(query, documents, categories)
    for score, content in results[:3]:  # Display top 3 results
        print(f"Score: {score:.4f}")
        print(f"Content: {content}")
