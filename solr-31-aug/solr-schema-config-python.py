import pysolr
import json
import requests

def configure_solr_schema(solr_url):
    # Connect to Solr
    solr = pysolr.Solr(solr_url, timeout=10)
    
    # Define the schema fields
    schema_fields = [
        {
            "name": "id",
            "type": "string",
            "indexed": True,
            "stored": True,
            "required": True,
            "multiValued": False
        },
        {
            "name": "page_title",
            "type": "text_general",
            "indexed": True,
            "stored": True
        },
        {
            "name": "category",
            "type": "string",
            "indexed": True,
            "stored": True
        },
        {
            "name": "embedding",
            "type": "knn_vector",
            "indexed": True,
            "stored": True,
            "vectorDimension": 1024
        }
    ]
    
    # Define the field types
    field_types = [
        {
            "name": "knn_vector",
            "class": "solr.DenseVectorField",
            "vectorDimension": 1024,
            "similarityFunction": "cosine"
        }
    ]
    
    # Add fields to the schema
    for field in schema_fields:
        try:
            response = requests.post(f"{solr_url}/schema", json={
                "add-field": field
            })
            response.raise_for_status()
            print(f"Added field: {field['name']}")
        except requests.exceptions.RequestException as e:
            print(f"Error adding field {field['name']}: {e}")
    
    # Add field types to the schema
    for field_type in field_types:
        try:
            response = requests.post(f"{solr_url}/schema", json={
                "add-field-type": field_type
            })
            response.raise_for_status()
            print(f"Added field type: {field_type['name']}")
        except requests.exceptions.RequestException as e:
            print(f"Error adding field type {field_type['name']}: {e}")

# Usage
solr_url = 'http://localhost:8983/solr/bank_website'
configure_solr_schema(solr_url)
