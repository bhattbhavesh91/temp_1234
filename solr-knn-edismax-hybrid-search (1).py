import pysolr
from sentence_transformers import SentenceTransformer

class SolrHybridSearch:
    def __init__(self, solr_url, embedding_model='all-MiniLM-L6-v2'):
        self.solr = pysolr.Solr(solr_url, always_commit=True)
        self.embedding_model = SentenceTransformer(embedding_model)

    def generate_embedding(self, text):
        return self.embedding_model.encode([text])[0]

    def hybrid_search(self, query, filter_queries=None, num_results=30, knn_top_k=10, rerank_docs=50, rerank_weight=3):
        # Generate query embedding
        query_embedding = self.generate_embedding(query)

        # Prepare the Solr query for each vector field
        vector_fields = ['page_title_vector', 'page_content_vector', 'tags_vector', 'page_description_vector']
        knn_queries = [f"{{!knn f={field} topK={knn_top_k}}}{str(query_embedding.tolist())}" for field in vector_fields]

        # Combine kNN queries with OR
        combined_knn_query = " OR ".join(knn_queries)

        # Prepare the full Solr query
        solr_query = {
            'fl': ['id', 'page_title', 'page_content', 'tags', 'page_description', 'score'],
            'q': combined_knn_query,
            'rq': f'{{!rerank reRankQuery=$rqq reRankDocs={rerank_docs} reRankWeight={rerank_weight}}}',
            'rqq': f"{{!edismax qf='page_title^5 page_content^2 tags^3 page_description^1'}}{query}",
            'rows': num_results
        }

        # Add filter queries if provided
        if filter_queries:
            solr_query['fq'] = filter_queries

        # Execute the search
        solr_response = self.solr.search(**solr_query)

        # Process and return the results
        results = []
        for doc in solr_response:
            results.append({
                'id': doc['id'],
                'page_title': doc.get('page_title', ''),
                'page_content': doc.get('page_content', ''),
                'tags': doc.get('tags', []),
                'page_description': doc.get('page_description', ''),
                'score': doc.get('score', 0)
            })

        return results

# Usage example
if __name__ == "__main__":
    solr_url = 'http://localhost:8983/solr/bank_core'
    searcher = SolrHybridSearch(solr_url)

    query = "savings account interest rates"
    filter_queries = ['type:account']  # Example filter query
    results = searcher.hybrid_search(query, filter_queries=filter_queries)

    for result in results:
        print(f"Title: {result['page_title']}")
        print(f"Description: {result['page_description']}")
        print(f"Tags: {', '.join(result['tags'])}")
        print(f"Score: {result['score']}")
        print("---")
