# inference.py
from transformers import AutoTokenizer, AutoModel
import torch

def model_fn(model_dir):
    """Load the model for inference."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    return model, tokenizer

def input_fn(input_data, content_type='application/json'):
    """Preprocess the input data."""
    if content_type == 'application/json':
        import json
        data = json.loads(input_data)
        return data['inputs']
    else:
        raise ValueError("Unsupported content type: {}".format(content_type))

def predict_fn(input_data, model_and_tokenizer):
    """Make a prediction."""
    model, tokenizer = model_and_tokenizer
    inputs = tokenizer(input_data, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state
    return embeddings.mean(dim=1).tolist()

def output_fn(prediction, accept='application/json'):
    """Format the prediction output."""
    if accept == 'application/json':
        import json
        return json.dumps({'embeddings': prediction})
    else:
        raise ValueError("Unsupported accept type: {}".format(accept))
