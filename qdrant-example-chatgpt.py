from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import boto3
import json

# Initialize Qdrant client
qdrant_client = QdrantClient(host="localhost", port=6333)

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# AWS Bedrock setup
bedrock_client = boto3.client(
    service_name="bedrock-runtime", 
    region_name="us-east-1"  # Adjust this to your AWS region
)

# Qdrant Collection Configuration
COLLECTION_NAME = "page_index"
VECTOR_DIM = 384  # Dimension of the embedding model

def create_qdrant_collection():
    """Create a Qdrant collection for storing vectors."""
    try:
        qdrant_client.get_collection(COLLECTION_NAME)
        print(f"Collection {COLLECTION_NAME} already exists!")
    except:
        # Create a new collection
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        print(f"Collection {COLLECTION_NAME} created successfully!")

def add_document(doc_id, title, content):
    """Add a document to Qdrant."""
    content_embedding = embedding_model.encode(content).tolist()
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=doc_id,
                vector=content_embedding,
                payload={"title": title, "content": content},
            )
        ],
    )

def search_qdrant(query, top_n=3):
    """Perform hybrid search (vector similarity + metadata) in Qdrant."""
    # Encode query to embedding
    query_embedding = embedding_model.encode(query).tolist()

    # Perform vector search
    search_results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_n,
    )

    # Extract results
    top_results = [
        {
            "title": result.payload["title"],
            "content": result.payload["content"],
        }
        for result in search_results
    ]
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
    # Step 1: Retrieve top N documents based on semantic search
    top_docs = search_qdrant(query, top_n=3)

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
    # Step 1: Create the Qdrant collection
    create_qdrant_collection()

    # Step 2: Add some documents
    add_document(
        doc_id=1,
        title="Fixed Deposit Interest Rates",
        content="The fixed deposit interest rate ranges from 5% to 7% annually.",
    )
    add_document(
        doc_id=2,
        title="Mayura Credit Card",
        content="The Mayura Credit Card offers 5% cashback on groceries and fuel.",
    )
    add_document(
        doc_id=3,
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
