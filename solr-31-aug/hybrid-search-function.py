import pysolr
from sentence_transformers import SentenceTransformer

def hybrid_search(query, solr_url, model_name='Alibaba-NLP/gte-large-en-v1.5', rows=10):
    # Initialize Solr connection
    solr = pysolr.Solr(solr_url, timeout=10)
    
    # Load the sentence transformer model
    model = SentenceTransformer(model_name)
    
    # Generate embedding for the query
    query_embedding = model.encode(query).tolist()
    
    # Prepare Solr query
    solr_query = {
        'q': query,
        'defType': 'edismax',
        'qf': 'page_title^2.0 category^1.5',
        'mm': '2<-1 5<-2 6<90%',
        'tie': 0.1,
        'vector': query_embedding,
        'rows': rows,
    }
    
    # Perform the search
    results = solr.search('*:*', **solr_query, handler='/hybrid_search')
    
    return results

# Example usage
solr_url = 'http://localhost:8983/solr/bank_website'
search_results = hybrid_search("home loan options", solr_url)

for result in search_results:
    print(f"ID: {result['id']}, Title: {result['page_title']}, Score: {result['score']}")
