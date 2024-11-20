import redis
import numpy as np
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer
import boto3
import json

# Initialize Redis connection
redis_client = redis.Redis(host="localhost", port=6379)

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# AWS Bedrock setup
bedrock_client = boto3.client(
    service_name="bedrock-runtime", 
    region_name="us-east-1"  # Adjust this to your AWS region
)

# Index configuration
VECTOR_DIM = 384  # Dimension of embedding model
INDEX_NAME = "page_index"
VECTOR_FIELD = "content_vector"

def create_index():
    """Create the Redis index for hybrid search."""
    try:
        # Check if the index already exists
        redis_client.ft(INDEX_NAME).info()
        print(f"Index {INDEX_NAME} already exists!")
        return
    except:
        pass  # Index doesn't exist

    # Define fields for Redis index
    schema = (
        TextField("title"),  # Page title for text-based search
        TextField("content"),  # Page content for retrieval
        VectorField(
            VECTOR_FIELD,
            "FLAT",  # Flat index for vector similarity
            {"TYPE": "FLOAT32", "DIM": VECTOR_DIM, "DISTANCE_METRIC": "COSINE"},
        ),
    )

    # Create the index
    redis_client.ft(INDEX_NAME).create_index(
        schema,
        definition=IndexDefinition(prefix=["page:"], index_type=IndexType.HASH),
    )
    print(f"Index {INDEX_NAME} created successfully!")

def add_document(doc_id, title, content):
    """Add a document to the Redis index."""
    content_embedding = embedding_model.encode(content).astype("float32").tobytes()
    redis_client.hset(
        f"page:{doc_id}",
        mapping={
            "title": title,
            "content": content,
            VECTOR_FIELD: content_embedding,
        },
    )

def search_redis(query, top_n=3):
    """Perform hybrid search (text and vector) in Redis."""
    # Encode query to embedding
    query_embedding = embedding_model.encode(query).astype("float32").tobytes()

    # Create a hybrid query (semantic + full-text search)
    hybrid_query = (
        Query(f"@title|@content:{query}")  # Full-text search
        .return_fields("title", "content", VECTOR_FIELD)
        .sort_by(f"__vector_score__", asc=False)
        .paging(0, top_n)
        .dialect(2)
    )

    # Add vector similarity clause
    hybrid_query.vector_query(
        VECTOR_FIELD,
        query_embedding,
        "KNN",  # Approximate Nearest Neighbors search
        K=top_n,
    )

    # Execute query
    results = redis_client.ft(INDEX_NAME).search(hybrid_query)

    # Parse results
    top_results = []
    for doc in results.docs:
        top_results.append({"title": doc.title, "content": doc.content})
    return top_results

def query_bedrock(model_id, prompt):
    """Query the Amazon Bedrock model."""
    response = bedrock_client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": prompt}),
    )
    response_body = json.loads(response["body"])
    return response_body.get("result", "")

def hybrid_rag_solution(query, model_id):
    """Full Hybrid RAG pipeline."""
    # Step 1: Retrieve top N documents based on hybrid search
    top_docs = search_redis(query, top_n=3)

    # Step 2: Format the retrieved content for the LLM
    context = "\n\n".join(
        [f"Title: {doc['title']}\nContent: {doc['content']}" for doc in top_docs]
    )
    llm_prompt = (
        f"Answer the question based on the following context:\n\n{context}\n\n"
        f"Question: {query}\nAnswer:"
    )

    # Step 3: Query the Bedrock model for a response
    answer = query_bedrock(model_id, llm_prompt)

    # Return the response with sources
    return {
        "answer": answer,
        "sources": [{"title": doc["title"], "content": doc["content"]} for doc in top_docs],
    }

# Example Usage
if __name__ == "__main__":
    # Step 1: Create the index
    create_index()

    # Step 2: Add some documents
    add_document(
        doc_id="1",
        title="Fixed Deposit Interest Rates",
        content="The fixed deposit interest rate ranges from 5% to 7% annually.",
    )
    add_document(
        doc_id="2",
        title="Mayura Credit Card",
        content="The Mayura Credit Card offers 5% cashback on groceries and fuel.",
    )
    add_document(
        doc_id="3",
        title="Home Loan Options",
        content="Home loans are available with interest rates starting from 6.5%.",
    )

    # Step 3: Query the RAG system
    user_query = "What are the benefits of the Mayura Credit Card?"
    model_id = "amazon.com.titan.text"  # Replace with your Bedrock model ID
    response = hybrid_rag_solution(user_query, model_id)

    # Print the response
    print("Answer:", response["answer"])
    print("Sources:", response["sources"])
