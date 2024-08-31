import pysolr
from sentence_transformers import SentenceTransformer
import json
import numpy as np

# Initialize Solr connection
solr = pysolr.Solr('http://localhost:8983/solr/banking_core', always_commit=True)

# Load the sentence transformer model for query embedding
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

def expand_query(query):
    expanded = query
    for category, terms in banking_terms.items():
        if any(term in query.lower() for term in terms):
            expanded += f" {category} {' '.join(terms)}"
    return expanded

def advanced_banking_search(query, num_results=10, use_embeddings=False, embedding_field=None):
    expanded_query = expand_query(query)
    query_embedding = model.encode(expanded_query).tolist()

    # Base Solr query
    solr_query = {
        'query': expanded_query,
        'fields': '*,score',
        'rows': num_results,
        'defType': 'edismax',
        'qf': 'content^2.0 category^1.5',
        'pf': 'content^1.5',
        'mm': '2<-1 5<-2 6<90%',
        'tie': 0.1,
        'bf': [
            'termfreq(content,{})^0.3'.format(expanded_query),
            'log(pageviews)^0.1'
        ],
        'fl': '*,score,termfreq(content,{}) as tf_score'.format(expanded_query),
        'sort': 'score desc',
        'debug': 'true'
    }

    # Add embedding-based search if enabled and field is provided
    if use_embeddings and embedding_field:
        solr_query['bq'] = f'{{!func}}product({embedding_field},vector({json.dumps(query_embedding)}))^0.5'
        solr_query['fl'] += f',product({embedding_field},vector({json.dumps(query_embedding)})) as embedding_score'

    # Execute Solr query
    results = solr.search(**solr_query)
    
    # If embeddings are not stored in Solr, compute similarity here
    if use_embeddings and not embedding_field:
        for result in results:
            content_embedding = model.encode(result['content'])
            result['embedding_score'] = np.dot(query_embedding, content_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(content_embedding))
        
        # Re-sort results based on combined score
        results = sorted(results, key=lambda x: x['score'] + x.get('embedding_score', 0), reverse=True)

    return results[:num_results]

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

# Set these based on your Solr setup
use_embeddings = True  # Set to False if you don't want to use embeddings
embedding_field = 'embedding_vector'  # Set to None if embeddings are not stored in Solr

for query in queries:
    print(f"\nQuery: {query}")
    results = advanced_banking_search(query, use_embeddings=use_embeddings, embedding_field=embedding_field)
    for result in results[:3]:  # Display top 3 results
        print(f"Score: {result['score']:.4f}")
        print(f"TF Score: {result.get('tf_score', 'N/A')}")
        print(f"Embedding Score: {result.get('embedding_score', 'N/A')}")
        print(f"Content: {result['content']}")
        print(f"Category: {result.get('category', 'N/A')}")
        print()
