import redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
import numpy as np
from sentence_transformers import SentenceTransformer
import json

class DocumentRetrieval:
    def __init__(self, host='localhost', port=6379, db=0):
        # Initialize Redis connection
        self.redis_client = redis.Redis(host=host, port=port, db=db)
        # Initialize sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # Vector dimension from the model
        self.vector_dim = 384
        # Initialize indexes
        self._create_indexes()
    
    def _create_indexes(self):
        """Create separate indexes for titles and content"""
        try:
            # Create title index with document mapping
            title_schema = (
                TextField("$.title", as_name="title"),
                TextField("$.doc_id", as_name="doc_id"),
                TextField("$.content_chunks", as_name="content_chunks")  # Store chunk IDs
            )
            
            self.redis_client.ft("title_idx").create_index(
                title_schema,
                definition=IndexDefinition(
                    prefix=["title:"],
                    index_type=IndexType.JSON
                )
            )
        except redis.exceptions.ResponseError as e:
            if "Index already exists" not in str(e):
                raise e

        try:
            # Create content index with vector search capabilities
            content_schema = (
                TextField("$.content", as_name="content"),
                TextField("$.doc_id", as_name="doc_id"),
                VectorField("$.embedding", 
                           "HNSW", 
                           {
                               "TYPE": "FLOAT32",
                               "DIM": self.vector_dim,
                               "DISTANCE_METRIC": "COSINE"
                           }, 
                           as_name="embedding")
            )
            
            self.redis_client.ft("content_idx").create_index(
                content_schema,
                definition=IndexDefinition(
                    prefix=["content:"],
                    index_type=IndexType.JSON
                )
            )
        except redis.exceptions.ResponseError as e:
            if "Index already exists" not in str(e):
                raise e

    def chunk_text(self, text, chunk_size=1000, overlap=100):
        """Split text into chunks with overlap"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            
        return chunks

    def index_document(self, doc_id, title, content):
        """Index a document's title and content with linked structure"""
        # Process content in chunks first
        chunks = self.chunk_text(content)
        chunk_ids = []
        
        # Index each chunk with its embedding
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_ids.append(chunk_id)
            embedding = self.model.encode(chunk)
            
            content_data = {
                "content": chunk,
                "doc_id": doc_id,
                "embedding": embedding.tolist()
            }
            
            self.redis_client.json().set(f"content:{chunk_id}", "$", content_data)
        
        # Index title with reference to content chunks
        title_data = {
            "title": title,
            "doc_id": doc_id,
            "content_chunks": chunk_ids  # Store references to chunks
        }
        self.redis_client.json().set(f"title:{doc_id}", "$", title_data)

    def search(self, query, title_filter_threshold=5, final_top_k=5):
        """
        Sequential search:
        1. First filter by title keywords
        2. Then perform vector similarity search on content of filtered documents
        """
        # Step 1: Filter by title keywords
        title_query = Query(f"@title:({query})").return_fields("doc_id", "content_chunks").paging(0, title_filter_threshold)
        title_results = self.redis_client.ft("title_idx").search(title_query)
        
        if len(title_results.docs) == 0:
            return []
        
        # Collect all chunk IDs from filtered documents
        relevant_chunks = []
        doc_chunk_mapping = {}  # Keep track of which chunks belong to which document
        
        for doc in title_results.docs:
            doc_dict = json.loads(doc.json)
            doc_id = doc_dict['doc_id']
            chunk_ids = doc_dict['content_chunks']
            relevant_chunks.extend(chunk_ids)
            
            # Map chunks to their document
            for chunk_id in chunk_ids:
                doc_chunk_mapping[chunk_id] = doc_id
        
        if not relevant_chunks:
            return []
        
        # Step 2: Perform vector similarity search on filtered chunks
        query_embedding = self.model.encode(query)
        
        # Create a filtered query for specific chunk IDs
        chunk_filter = '|'.join([f'@doc_id:({chunk_id})' for chunk_id in relevant_chunks])
        content_query = (
            Query(f'({chunk_filter})=>[KNN {final_top_k} @embedding $query_vector AS score]')
            .dialect(2)
            .return_fields("doc_id", "content", "score")
            .paging(0, final_top_k)
        )
        
        content_results = self.redis_client.ft("content_idx").search(
            content_query,
            query_params={
                'query_vector': np.array(query_embedding, dtype=np.float32).tobytes()
            }
        )
        
        # Process results
        results = []
        seen_docs = set()
        
        for doc in content_results.docs:
            doc_dict = json.loads(doc.json)
            chunk_id = doc_dict['doc_id']
            original_doc_id = doc_chunk_mapping[chunk_id]
            
            # Skip if we already have a result from this document
            if original_doc_id in seen_docs:
                continue
                
            seen_docs.add(original_doc_id)
            
            # Get original document title
            title_data = json.loads(self.redis_client.json().get(f"title:{original_doc_id}"))
            
            results.append({
                'doc_id': original_doc_id,
                'title': title_data['title'],
                'content': doc_dict['content'],
                'score': float(doc_dict.get('score', 0))
            })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)

# Example usage
if __name__ == "__main__":
    # Initialize the retrieval system
    retrieval = DocumentRetrieval()
    
    # Example documents
    documents = [
        {
            "id": "doc1",
            "title": "Introduction to Machine Learning",
            "content": "Machine learning is a subset of artificial intelligence..."
        },
        {
            "id": "doc2",
            "title": "Python Programming Basics",
            "content": "Python is a high-level programming language..."
        }
    ]
    
    # Index documents
    for doc in documents:
        retrieval.index_document(doc["id"], doc["title"], doc["content"])
    
    # Search example
    results = retrieval.search("machine learning concepts")
    
    # Print results
    for result in results:
        print(f"Document ID: {result['doc_id']}")
        print(f"Title: {result['title']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Content snippet: {result['content'][:200]}...")
        print("-" * 80)
