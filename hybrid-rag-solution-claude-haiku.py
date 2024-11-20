import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import openai

class HybridRAGSystem:
    def __init__(self, 
                 qdrant_host: str = 'localhost', 
                 qdrant_port: int = 6333,
                 openai_model: str = 'gpt-3.5-turbo'):
        # Vector Embedding Model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Qdrant Client
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        
        # Collection name
        self.collection_name = 'website_pages'
        
        # OpenAI Configuration
        openai.api_key = 'your_openai_key_here'
        self.llm_model = openai_model

    def create_collection(self, vector_size: int = 384):
        """Create Qdrant collection for storing page embeddings."""
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "size": vector_size,
                "distance": "cosine"
            }
        )

    def insert_documents(self, documents: List[Dict[str, Any]]):
        """
        Insert documents into Qdrant with title and content embeddings.
        
        Expected document structure:
        {
            'title': 'Page Title',
            'content': 'Page Content'
        }
        """
        points = []
        for idx, doc in enumerate(documents):
            # Generate embeddings for title and content
            title_embedding = self.embedding_model.encode(doc['title']).tolist()
            content_embedding = self.embedding_model.encode(doc['content']).tolist()
            
            points.append({
                'id': idx,
                'vector': {
                    'title_vector': title_embedding,
                    'content_vector': content_embedding
                },
                'payload': doc
            })
        
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def hybrid_search(self, query: str, top_k: int = 3):
        """
        Perform hybrid search across title and content embeddings.
        
        Returns top results with both text and embedding-based search.
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Perform title vector search
        title_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=('title_vector', query_embedding),
            limit=top_k
        )
        
        # Perform content vector search
        content_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=('content_vector', query_embedding),
            limit=top_k
        )
        
        # Combine and deduplicate results
        all_results = title_results + content_results
        unique_results = {result.id: result for result in all_results}
        
        return list(unique_results.values())[:top_k]

    def generate_answer(self, query: str, context: List[Dict]) -> str:
        """Generate answer using OpenAI LLM with retrieved context."""
        context_str = "\n\n".join([
            f"Title: {item.payload['title']}\nContent: {item.payload['content']}"
            for item in context
        ])
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuery: {query}\n\nAnswer the query using only the provided context."}
        ]
        
        response = openai.ChatCompletion.create(
            model=self.llm_model,
            messages=messages
        )
        
        return response.choices[0].message.content

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Complete query processing pipeline:
        1. Hybrid search
        2. Context retrieval
        3. Answer generation
        """
        # Retrieve context
        context = self.hybrid_search(query)
        
        # Generate answer
        answer = self.generate_answer(query, context)
        
        return {
            'answer': answer,
            'sources': [
                {'title': result.payload['title'], 'score': result.score}
                for result in context
            ]
        }

# Usage Example
def main():
    # Initialize RAG System
    rag_system = HybridRAGSystem()
    
    # Create Collection
    rag_system.create_collection()
    
    # Sample Documents
    documents = [
        {
            'title': 'Fixed Deposit Interest Rates',
            'content': 'Our bank offers competitive fixed deposit rates ranging from 4.5% to 7.2% depending on tenure and deposit amount.'
        },
        {
            'title': 'Mayura Credit Card',
            'content': 'Mayura Credit Card provides cashback of 2% on all grocery purchases and travel rewards with no annual fees.'
        }
    ]
    
    # Insert Documents
    rag_system.insert_documents(documents)
    
    # Process Query
    result = rag_system.process_query("What are the interest rates for fixed deposits?")
    print(result)

if __name__ == "__main__":
    main()
