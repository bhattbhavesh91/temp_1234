import pysolr
from sentence_transformers import SentenceTransformer
import json

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

def advanced_banking_search(query, num_results=10):
    expanded_query = expand_query(query)
    query_embedding = model.encode(expanded_query).tolist()

    # Construct Solr query
    solr_query = {
        'query': expanded_query,
        'fields': '*,score',
        'rows': num_results,
        'defType': 'edismax',
        'qf': 'content^2.0 category^1.5',
        'pf': 'content^1.5',  # phrase fields for better text matching
        'mm': '2<-1 5<-2 6<90%',  # minimum should match for multi-term queries
        'tie': 0.1,  # tiebreaker for multiple fields
        'bq': [
            f'{{!func}}product(payload(embedding_payload),vector({json.dumps(query_embedding)}))^0.5',
            '{!func}recip(ms(NOW,last_updated),3.16e-11,1,1)^0.3'  # recency boost
        ],
        'bf': [
            'termfreq(content,{})^0.3'.format(expanded_query),  # boost based on term frequency
            'log(pageviews)^0.1'  # example of boosting by popularity
        ],
        'fl': '*,score,termfreq(content,{}) as tf_score,product(payload(embedding_payload),vector({})) as embedding_score'.format(
            expanded_query, json.dumps(query_embedding)),
        'sort': 'score desc',
        'debug': 'true'  # This will return detailed scoring information
    }

    # Execute Solr query
    results = solr.search(**solr_query)
    
    return results

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
    for result in results[:3]:  # Display top 3 results
        print(f"Score: {result['score']:.4f}")
        print(f"TF Score: {result.get('tf_score', 'N/A')}")
        print(f"Embedding Score: {result.get('embedding_score', 'N/A')}")
        print(f"Content: {result['content']}")
        print(f"Category: {result.get('category', 'N/A')}")
        print(f"Debug: {result.get('debug', 'N/A')}")
        print()
