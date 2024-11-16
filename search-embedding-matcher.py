import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from typing import List, Tuple, Dict
import re

class SearchQueryMatcher:
    def __init__(self, model_name: str = "thenlper/gte-large"):
        """
        Initialize the matcher with the GTE-large model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.url_embeddings = None
        self.urls = None
        self.query_clusters = {}
        
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
        
    def create_query_clusters(self, df: pd.DataFrame, tag_column: str) -> Dict[str, List[str]]:
        """
        Group similar search queries together based on their embeddings.
        
        Args:
            df: DataFrame containing search tags and URLs
            tag_column: Name of the column containing search tags
        """
        # Split the tags and create a list of all unique queries
        all_queries = []
        for tags in df[tag_column]:
            queries = [self.preprocess_text(q.strip()) for q in tags.split('|')]
            all_queries.extend(queries)
        unique_queries = list(set(all_queries))
        
        # Get embeddings for all unique queries
        query_embeddings = self.model.encode(unique_queries, convert_to_tensor=True)
        
        # Cluster similar queries using cosine similarity
        similarity_matrix = cosine_similarity(query_embeddings.cpu().numpy())
        
        # Group similar queries (similarity > 0.85)
        processed_queries = set()
        for i, query in enumerate(unique_queries):
            if query not in processed_queries:
                similar_queries = []
                for j, sim_score in enumerate(similarity_matrix[i]):
                    if sim_score > 0.85:  # Threshold for similarity
                        similar_queries.append(unique_queries[j])
                        processed_queries.add(unique_queries[j])
                        
                if similar_queries:
                    self.query_clusters[query] = similar_queries
                    
        return self.query_clusters
    
    def fit(self, df: pd.DataFrame, tag_column: str, url_column: str):
        """
        Fit the matcher with the dataset.
        
        Args:
            df: DataFrame containing search tags and URLs
            tag_column: Name of the column containing search tags
            url_column: Name of the column containing URLs
        """
        # Create query clusters
        self.create_query_clusters(df, tag_column)
        
        # Create a mapping of representative queries to URLs
        query_url_map = {}
        for _, row in df.iterrows():
            tags = [self.preprocess_text(tag.strip()) for tag in row[tag_column].split('|')]
            url = row[url_column]
            
            # Find the representative query for these tags
            for tag in tags:
                for rep_query, similar_queries in self.query_clusters.items():
                    if tag in similar_queries:
                        if rep_query not in query_url_map:
                            query_url_map[rep_query] = set()
                        query_url_map[rep_query].add(url)
        
        # Create embeddings for representative queries
        rep_queries = list(query_url_map.keys())
        self.query_embeddings = self.model.encode(rep_queries, convert_to_tensor=True)
        
        # Store URLs and their mapping
        self.urls = rep_queries
        self.url_map = query_url_map
        
    def find_matches(self, query: str, top_k: int = 5) -> List[Tuple[str, float, List[str]]]:
        """
        Find matching URLs for a given search query.
        
        Args:
            query: Search query
            top_k: Number of top matches to return
            
        Returns:
            List of tuples containing (representative_query, similarity_score, urls)
        """
        # Preprocess and embed the query
        query = self.preprocess_text(query)
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Calculate similarities
        similarities = cosine_similarity(
            query_embedding.cpu().numpy().reshape(1, -1),
            self.query_embeddings.cpu().numpy()
        )[0]
        
        # Get top-k matches
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            rep_query = self.urls[idx]
            score = similarities[idx]
            matched_urls = list(self.url_map[rep_query])
            results.append((rep_query, score, matched_urls))
            
        return results

# Example usage
if __name__ == "__main__":
    # Sample data
    data = {
        'search_tags': [
            'block credit card|credit card block|how to block credit card',
            'credit card application|apply for credit card|credit card apply',
            'credit card benefits|credit card rewards|card benefits'
        ],
        'urls': [
            'example.com/block-card',
            'example.com/apply',
            'example.com/benefits'
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Initialize and fit the matcher
    matcher = SearchQueryMatcher()
    matcher.fit(df, 'search_tags', 'urls')
    
    # Find matches for a query
    test_query = "I want to block my credit card"
    matches = matcher.find_matches(test_query, top_k=3)
    
    for query, score, urls in matches:
        print(f"Matched Query: {query}")
        print(f"Similarity Score: {score:.3f}")
        print(f"URLs: {urls}\n")
