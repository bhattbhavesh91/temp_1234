import pysolr
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

# Initialize Solr connection
solr = pysolr.Solr('http://localhost:8983/solr/banking_core', always_commit=True)

# Load the sentence transformer model
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

def preprocess(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    return ' '.join([w for w in tokens if w not in stop_words])

def expand_query(query):
    expanded = query
    for category, terms in banking_terms.items():
        if any(term in query.lower() for term in terms):
            expanded += f" {category} {' '.join(terms)}"
    return expanded

def custom_ranking(query, results):
    query_embedding = model.encode(preprocess(query))
    
    # Prepare TF-IDF vectorizer
    tfidf = TfidfVectorizer()
    all_texts = [result['content'] for result in results] + [query]
    tfidf.fit(all_texts)
    
    # Calculate custom scores
    custom_scores = []
    for result in results:
        content = result['content']
        content_embedding = np.array(result['embedding'])
        
        # Cosine similarity between query and content embeddings
        cosine_sim = np.dot(query_embedding, content_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(content_embedding))
        
        # TF-IDF similarity
        query_tfidf = tfidf.transform([query]).toarray()
        content_tfidf = tfidf.transform([content]).toarray()
        tfidf_sim = np.dot(query_tfidf, content_tfidf.T)[0][0]
        
        # Combine similarities
        custom_score = 0.7 * cosine_sim + 0.3 * tfidf_sim
        custom_scores.append((custom_score, result))
    
    # Sort results by custom score
    return sorted(custom_scores, key=lambda x: x[0], reverse=True)

def advanced_banking_search(query, num_results=10):
    expanded_query = expand_query(query)
    
    # Perform initial Solr query
    solr_results = solr.search(expanded_query, **{
        'fl': '*,score',  # Return all fields and the Solr score
        'rows': num_results * 2  # Retrieve more results than needed for re-ranking
    })
    
    # Convert Solr results to a list of dictionaries
    results = [dict(result) for result in solr_results]
    
    # Apply custom ranking
    ranked_results = custom_ranking(query, results)
    
    return ranked_results[:num_results]

# Example usage
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
    results = advanced_banking_search(query)
    for score, result in results[:3]:  # Display top 3 results
        print(f"Score: {score:.4f}")
        print(f"Content: {result['content']}")
        print(f"Category: {result.get('category', 'N/A')}")
        print()
