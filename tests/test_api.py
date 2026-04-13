import pytest
from unittest.mock import patch
from src.serving.app import app
from src.schemas.classification import (
    ClassificationResult,
    BusinessCategory,
    NamedEntity,
)


@pytest.fixture
def client():
    # Creates a test version of the Flask app — no real server needed
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def mock_classification_result():
    # A fake result we control — no API call made
    return ClassificationResult(
        business=BusinessCategory.ECONOMY,
        confidence=0.95,
        named_entities=[NamedEntity(name="Tony Blair", job="Prime Minister")],
        april_events=[]
    )


@patch("src.serving.app.create_table")
def test_health_endpoint(mock_create_table, client):
    # Check the health endpoint returns ok
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


@patch("src.serving.app.create_table")
@patch("src.serving.app.save_classification")
def test_classify_returns_200(mock_save, mock_create_table, client):
    # Check the classify endpoint returns 200 with valid input
    with patch("src.serving.app.classifier.classify", return_value=mock_classification_result()):
        response = client.post(
            "/classify",
            json={"text": "The economy grew by 3% last April according to the Chancellor."}
        )
        assert response.status_code == 200


@patch("src.serving.app.create_table")
@patch("src.serving.app.save_classification")
def test_classify_returns_correct_fields(mock_save, mock_create_table, client):
    # Check the response contains all expected fields
    with patch("src.serving.app.classifier.classify", return_value=mock_classification_result()):
        response = client.post(
            "/classify",
            json={"text": "The economy grew by 3% last April according to the Chancellor."}
        )
        data = response.get_json()
        assert "business" in data
        assert "confidence" in data
        assert "named_entities" in data
        assert "april_events" in data


@patch("src.serving.app.create_table")
def test_classify_rejects_missing_text(mock_create_table, client):
    # Check the API rejects a request with no text field
    response = client.post("/classify", json={})
    assert response.status_code == 400


@patch("src.serving.app.create_table")
def test_classify_rejects_empty_text(mock_create_table, client):
    # Check the API rejects a request with empty text
    response = client.post("/classify", json={"text": ""})
    assert response.status_code == 400


@patch("src.serving.app.create_table")
def test_history_returns_list(mock_create_table, client):
    # Mock the database connection where it actually lives — in src.database.db
    with patch("src.database.db.get_connection") as mock_conn:
        mock_cursor = mock_conn.return_value.cursor.return_value
        mock_cursor.fetchall.return_value = []
        response = client.get("/history")
        assert response.status_code == 200
        assert isinstance(response.get_json(), list)