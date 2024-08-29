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
    # Use existing Solr configuration (including boosting and edismax)
    results = solr.search(query, **{
        'defType': 'edismax',  # Ensure edismax is used
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
        solr_scores = [doc['score'] for doc in results]
        llm_rankings = llm_rank(query, results)
        
        for rank, (doc, solr_score, llm_rank) in enumerate(zip(results, solr_scores, llm_rankings)):
            dataset.append({
                'qid': qid,
                'doc_id': doc['id'],
                'solr_score': solr_score,
                'llm_rank': llm_rank,
                'page_title': len(doc['page_title']),
                'page_content': len(doc['page_content']),
                'tags': len(doc['tags']),
                'page_description': len(doc['page_description'])
            })
    
    return pd.DataFrame(dataset)

def train_model(df):
    X = df[['solr_score', 'page_title', 'page_content', 'tags', 'page_description']]
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

def analyze_feature_importance(model, feature_names):
    feature_importance = model.feature_importances_
    feature_importance_normalized = feature_importance / np.sum(feature_importance)
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance_normalized
    })
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
    
    print("Feature Importance:")
    print(feature_importance_df)
    
    return feature_importance_df

# Main execution
queries = ['bank loan', 'credit card', 'savings account', 'mortgage rates']
df = create_dataset(queries)
model = train_model(df)

feature_names = ['solr_score', 'page_title', 'page_content', 'tags', 'page_description']
feature_importance = analyze_feature_importance(model, feature_names)

# Evaluate the model
def apply_model_scores(X):
    return model.predict(X)

df['model_score'] = apply_model_scores(df[feature_names])

# Calculate Spearman's rank correlation coefficient
correlation, _ = spearmanr(-df['llm_rank'], df['model_score'])
print(f"Spearman's rank correlation between LLM ranking and model score: {correlation}")

# Calculate correlation between Solr score and LLM ranking
solr_correlation, _ = spearmanr(-df['llm_rank'], df['solr_score'])
print(f"Spearman's rank correlation between LLM ranking and Solr score: {solr_correlation}")

print("\nModel Performance:")
print(f"Model correlation: {correlation}")
print(f"Solr correlation: {solr_correlation}")
print(f"Improvement: {(correlation - solr_correlation) / solr_correlation * 100:.2f}%")
