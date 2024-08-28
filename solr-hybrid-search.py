import pysolr
import numpy as np
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

class SolrHybridSearch:
    def __init__(self, solr_url, embedding_model='all-MiniLM-L6-v2'):
        self.solr = pysolr.Solr(solr_url, always_commit=True)
        self.embedding_model = SentenceTransformer(embedding_model)

    def generate_embedding(self, text):
        return self.embedding_model.encode(text)

    def hybrid_search(self, query, num_results=10, edismax_weight=0.5, vector_weight=0.5):
        # Generate query embedding
        query_embedding = self.generate_embedding(query)

        # Perform eDismax search
        edismax_results = self.solr.search(query, **{
            'defType': 'edismax',
            'qf': 'page_title^2 page_content^1.5 tags^1.2 page_description^1',
            'rows': num_results * 2  # Fetch more results to re-rank
        })

        # Perform vector search and re-ranking
        vector_results = []
        for doc in edismax_results:
            # Combine embeddings from different fields
            doc_embedding = np.mean([
                np.array(doc.get('page_title_embedding', [0] * 768)),
                np.array(doc.get('page_content_embedding', [0] * 768)),
                np.array(doc.get('tags_embedding', [0] * 768)),
                np.array(doc.get('page_description_embedding', [0] * 768))
            ], axis=0)

            # Calculate cosine similarity
            similarity = 1 - cosine(query_embedding, doc_embedding)

            # Combine eDismax score and vector similarity
            combined_score = (edismax_weight * doc['score'] + 
                              vector_weight * similarity)

            vector_results.append((doc, combined_score))

        # Sort results by combined score
        sorted_results = sorted(vector_results, key=lambda x: x[1], reverse=True)

        # Return top results
        return [doc for doc, score in sorted_results[:num_results]]

    def search(self, query, num_results=10):
        results = self.hybrid_search(query, num_results)
        return [
            {
                'id': doc['id'],
                'page_title': doc['page_title'],
                'page_content': doc['page_content'],
                'tags': doc.get('tags', []),
                'page_description': doc.get('page_description', '')
            }
            for doc in results
        ]

# Usage example
if __name__ == "__main__":
    solr_url = 'http://localhost:8983/solr/bank_core'
    searcher = SolrHybridSearch(solr_url)

    query = "savings account interest rates"
    results = searcher.search(query)

    for result in results:
        print(f"Title: {result['page_title']}")
        print(f"Description: {result['page_description']}")
        print(f"Tags: {', '.join(result['tags'])}")
        print("---")
