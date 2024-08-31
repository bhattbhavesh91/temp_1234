import numpy as np
from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5')

def get_category_aware_embedding(content, category):
    # Generate embeddings for content and category separately
    content_embedding = model.encode(content)
    category_embedding = model.encode(category)
    
    # Concatenate the embeddings
    return np.concatenate([content_embedding, category_embedding])

# Example content
home_loan_content = "Our home loans offer competitive interest rates with flexible repayment options."
personal_loan_content = "Get a personal loan with quick approval and minimal documentation."

# Generate category-aware embeddings
home_loan_embedding = get_category_aware_embedding(home_loan_content, "home loan")
personal_loan_embedding = get_category_aware_embedding(personal_loan_content, "personal loan")

# Function to search
def search(query, embeddings, contents):
    query_embedding = get_category_aware_embedding(query, "loan")  # Generic category for query
    similarities = [np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb)) 
                    for emb in embeddings]
    return sorted(zip(similarities, contents), reverse=True)

# Example search
query = "What is the tenure for home loan?"
results = search(query, [home_loan_embedding, personal_loan_embedding], 
                 [home_loan_content, personal_loan_content])

for score, content in results:
    print(f"Score: {score:.4f}")
    print(f"Content: {content}\n")
