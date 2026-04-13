import pytest
from src.schemas.classification import (
    ClassificationResult,
    BusinessCategory,
    NamedEntity,
)


def test_valid_business_classification():
    result = ClassificationResult(
        business=BusinessCategory.ECONOMY,
        confidence=0.9,
        named_entities=[],
        april_events=[]
    )
    assert result.business == BusinessCategory.ECONOMY
    assert result.confidence == 0.9


def test_confidence_must_be_between_0_and_1():
    with pytest.raises(Exception):
        ClassificationResult(
            confidence=1.5,
            named_entities=[],
            april_events=[]
        )


def test_named_entity_requires_name_and_job():
    entity = NamedEntity(name="Tony Blair", job="Prime Minister")
    assert entity.name == "Tony Blair"
    assert entity.job == "Prime Minister"


def test_empty_classification_is_valid():
    result = ClassificationResult(
        confidence=0.5,
        named_entities=[],
        april_events=[]
    )
    assert result.business is None
    assert result.sports is None
    assert result.entertainment is None