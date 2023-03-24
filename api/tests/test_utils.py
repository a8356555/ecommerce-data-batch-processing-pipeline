import unittest
from elasticsearch import Elasticsearch
from utils import _vector_query, get_recommendation

# Connect to Elasticsearch instance
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

class TestRecommendations(unittest.TestCase):
    
    def test_vector_query_cosine(self):
        """
        Test cosine similarity query
        """
        query_vec = [1, 2, 3]
        country = "US"
        vector_field = "model_factor"
        q = _vector_query(query_vec, country, vector_field, similarity_type='cosine')
        
        expected_result = {
            "query": {
                "script_score": {
                    "query" : { 
                        "bool" : {
                            "filter" : {
                                    "term" : {
                                        "country" : country
                                    }
                                }
                        }
                    },
                    "script": {
                        "source": "doc['model_factor'].size() == 0 ? 0 : cosineSimilarity(params.vector, 'model_factor') + 1.0",
                        "params": {
                            "vector": query_vec
                        }
                    }
                }
            }
        }
        
        self.assertEqual(q, expected_result)
    
    def test_get_recommendation(self):
        """
        Test getting recommendations
        """
        target_item_id = "1234"
        rec_num = 10
        index = "products"
        vector_field = "model_factor"
        
        src, hits = get_recommendation(es, target_item_id, rec_num, index, vector_field)
        
        # Check that the returned source contains the same ID as the input item ID
        self.assertEqual(src['id'], target_item_id)
        
        # Check that the output list has length equal to rec_num
        self.assertEqual(len(hits), rec_num)
        
        # Check that the output list does not contain the input item ID
        self.assertNotIn(target_item_id, [h['_id'] for h in hits])