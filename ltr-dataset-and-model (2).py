import pysolr
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRanker
import openai
from scipy.stats import spearmanr

# Initialize Solr and OpenAI
solr = pysolr.Solr('http://localhost:8983/solr/your_collection')
openai.api_key = 'your_openai_api_key'

def get_solr_results(query, rows=50):
    results = solr.search(query, **{
        'defType': 'edismax',
        'qf': 'page_title^4 page_content^2 tags^1.5 page_description^1',
        'bq': 'page_title^2.0 tags^1.5',
        'fl': 'id,page_title,page_content,tags,page_description,score',
        'rows': rows
    })
    return list(results)

def llm_rank(query, documents):
    prompt = f"Rank the following documents based on their relevance to the query: '{query}'. Provide rankings as a list of numbers from 1 to {len(documents)}, where 1 is most relevant."
    for i, doc in enumerate(documents):
        prompt += f"\n{i+1}. Title: {doc['page_title']}\nDescription: {doc['page_description'][:100]}..."
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant skilled in ranking documents based on relevance to a query."},
            {"role": "user", "content": prompt}
        ]
    )
    
    rankings = [int(rank) for rank in response.choices[0].message['content'].split(',')]
    return rankings

def create_dataset(queries):
    dataset = []
    for qid, query in enumerate(queries):
        results = get_solr_results(query)
        bm25_scores = [doc['score'] for doc in results]
        llm_rankings = llm_rank(query, results)
        
        for rank, (doc, bm25_score, llm_rank) in enumerate(zip(results, bm25_scores, llm_rankings)):
            dataset.append({
                'qid': qid,
                'doc_id': doc['id'],
                'bm25_score': bm25_score,
                'llm_rank': llm_rank,
                'page_title': len(doc['page_title']),
                'page_content': len(doc['page_content']),
                'tags': len(doc['tags']),
                'page_description': len(doc['page_description'])
            })
    
    return pd.DataFrame(dataset)

def train_model(df):
    X = df[['page_title', 'page_content', 'tags', 'page_description']]
    y = -df['llm_rank']  # Negative because lower rank is better
    groups = df.groupby('qid').size().values

    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        X, y, groups, test_size=0.2, random_state=42
    )

    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type="gbdt",
        n_estimators=100,
        importance_type="gain",
    )

    model.fit(
        X_train, y_train,
        group=groups_train,
        eval_set=[(X_test, y_test)],
        eval_group=[groups_test],
        eval_at=[5, 10, 20],
        verbose=10
    )

    return model

def analyze_feature_importance(model, initial_boosts):
    feature_importance = model.feature_importances_
    feature_names = ['page_title', 'page_content', 'tags', 'page_description']
    
    # Normalize feature importances
    total_importance = sum(feature_importance)
    normalized_importance = feature_importance / total_importance
    
    # Calculate adjusted importance based on initial boosts
    adjusted_importance = {}
    for name, importance, boost in zip(feature_names, normalized_importance, initial_boosts):
        adjusted_importance[name] = importance / boost
    
    # Normalize adjusted importances
    total_adjusted = sum(adjusted_importance.values())
    for name in adjusted_importance:
        adjusted_importance[name] /= total_adjusted
    
    return {
        'raw_importance': dict(zip(feature_names, feature_importance)),
        'normalized_importance': dict(zip(feature_names, normalized_importance)),
        'adjusted_importance': adjusted_importance
    }

# Main execution
queries = ['bank loan', 'credit card', 'savings account', 'mortgage rates']
df = create_dataset(queries)
model = train_model(df)

# Initial boosts from Solr query
initial_boosts = [6.0, 2.0, 3.0, 1.0]  # page_title: 4+2, page_content: 2, tags: 1.5+1.5, page_description: 1

importance_analysis = analyze_feature_importance(model, initial_boosts)

print("Raw Feature Importance:", importance_analysis['raw_importance'])
print("Normalized Feature Importance:", importance_analysis['normalized_importance'])
print("Adjusted Feature Importance (accounting for initial boosts):", importance_analysis['adjusted_importance'])

# Evaluate the model
def apply_adjusted_boosts(row, adjusted_importance):
    return sum(row[field] * importance for field, importance in adjusted_importance.items())

df['model_score'] = df.apply(lambda row: apply_adjusted_boosts(row, importance_analysis['adjusted_importance']), axis=1)

# Calculate Spearman's rank correlation coefficient
correlation, _ = spearmanr(-df['llm_rank'], df['model_score'])
print(f"Spearman's rank correlation: {correlation}")
