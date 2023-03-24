import os
from fastapi import FastAPI
from models import RecommendRequest

from elasticsearch import Elasticsearch
from utils import get_recommendation

ES_HOST = os.env['ES_HOST']
ES_PORT = os.env['ES_PORT']
ES_USER = os.env['ES_USER']
ES_PASSWORD = os.env['ES_PASSWORD']
app = FastAPI()

def get_es():
    address = f'http://{ES_HOST}:{ES_PORT}'
    es = Elasticsearch(hosts=[address], http_auth=(ES_USER, ES_PASSWORD), verify_certs=False)
    return es

@app.post("/recommend_similar_product")
def recommend_similar_product(recommend_request: RecommendRequest):
    es = get_es()
    product, recs = get_recommendation(es, recommend_request.item_id, num=recommend_request.rec_num, index='products')
    recommendations = []
    for rec in recs:
        r = {}
        r['item_id'] = rec['_source']['item_id']
        r['description'] = rec['_source']['description']
        r['country'] = rec['_source']['country']
        r['score'] = rec['_score']
        recommendations.append(r)
    return recommendations

