from transformers import LongformerTokenizer, LongformerModel
import torch

def generate_long_text_embedding(long_text, model_name='allenai/longformer-base-4096', chunk_size=4096):
    """
    Generate a single embedding for a long text using the Longformer model.
    
    Args:
        long_text (str): The long text to be embedded.
        model_name (str): The name of the Longformer model to use. Default is 'allenai/longformer-base-4096'.
        chunk_size (int): The maximum number of tokens per chunk. Default is 4096.
    
    Returns:
        torch.Tensor: A single embedding representing the entire long text.
    """
    # Load the tokenizer and model
    tokenizer = LongformerTokenizer.from_pretrained(model_name)
    model = LongformerModel.from_pretrained(model_name)
    
    # Tokenize the input text in chunks
    tokens = tokenizer(long_text, return_tensors='pt', max_length=chunk_size, truncation=False, padding=False)['input_ids'].squeeze()
    
    # Split tokens into chunks
    chunks = tokens.split(chunk_size)
    
    # Initialize a list to hold chunk embeddings
    chunk_embeddings = []
    
    # Generate embeddings for each chunk
    for chunk in chunks:
        # Add batch dimension and attention mask
        inputs = {'input_ids': chunk.unsqueeze(0), 'attention_mask': torch.ones_like(chunk).unsqueeze(0)}
        with torch.no_grad():
            outputs = model(**inputs)
        # Use CLS token's embedding as the representation of the chunk
        chunk_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
        chunk_embeddings.append(chunk_embedding)
    
    # Convert list to tensor
    chunk_embeddings = torch.stack(chunk_embeddings)
    
    # Average the chunk embeddings to get a single embedding for the entire text
    document_embedding = torch.mean(chunk_embeddings, dim=0)
    
    return document_embedding

# Example usage:
long_text = "Your long 40,000-word text goes here..."
document_embedding = generate_long_text_embedding(long_text)

print("Document embedding shape:", document_embedding.shape)
