import os
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from main import app
from models import RecommendRequest


@pytest.fixture(scope="module")
def test_client():
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="function")
def mock_es(mocker):
    mock_es = MagicMock()
    mocker.patch("app.get_es", return_value=mock_es)
    return mock_es


def test_recommend_similar_product(mock_es, test_client):
    # Set up mock Elasticsearch response
    mock_es.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_source": {
                        "item_id": "123",
                        "description": "Product 1",
                        "country": "US"
                    },
                    "_score": 0.8
                },
                {
                    "_source": {
                        "item_id": "456",
                        "description": "Product 2",
                        "country": "US"
                    },
                    "_score": 0.6
                }
            ]
        }
    }

    recommend_request = RecommendRequest(item_id="789", rec_num=2)
    response = test_client.post("/recommend_similar_product", json=recommend_request.dict())

    assert response.status_code == 200
    assert len(response.json()) == 2
    assert response.json()[0]["item_id"] == "123"
    assert response.json()[0]["description"] == "Product 1"
    assert response.json()[0]["country"] == "US"
    assert response.json()[0]["score"] == 0.8
    assert response.json()[1]["item_id"] == "456"
    assert response.json()[1]["description"] == "Product 2"
    assert response.json()[1]["country"] == "US"
    assert response.json()[1]["score"] == 0.6