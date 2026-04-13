from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional


class BusinessCategory(str, Enum):
    STOCK_MARKET = "stock_market"
    COMPANY_NEWS = "company_news"
    ECONOMY = "economy"
    PERSONAL_FINANCE = "personal_finance"
    PROPERTY = "property"
    TECHNOLOGY_MEDIA = "technology_media"
    RETAIL_ONLINE_RETAIL = "retail_online_retail"
    SMALL_BUSINESS = "small_business"
    MERGERS_AND_ACQUISITIONS = "mergers_and_acquisitions"
    ENERGY_OIL_GAS = "energy_oil_gas"
    OTHER = "other"


class EntertainmentCategory(str, Enum):
    CINEMA = "cinema"
    THEATRE = "theatre"
    MUSIC = "music"
    LITERATURE = "literature"
    TELEVISION = "television"
    OTHER = "other"


class SportsCategory(str, Enum):
    FOOTBALL = "football"
    CRICKET = "cricket"
    RUGBY_UNION = "rugby_union"
    TENNIS = "tennis"
    GOLF = "golf"
    FORMULA_1 = "formula_1"
    ATHLETICS = "athletics"
    BOXING = "boxing"
    OTHER = "other"


class NamedEntity(BaseModel):
    name: str = Field(description="Full name of the person")
    job: str = Field(description="Their role or job title")


class EventSummary(BaseModel):
    event_date: str = Field(description="Date of the event in April")
    title: Optional[str] = None
    description: Optional[str] = None


class ClassificationResult(BaseModel):
    business: Optional[BusinessCategory] = None
    sports: Optional[SportsCategory] = None
    entertainment: Optional[EntertainmentCategory] = None
    confidence: float = Field(ge=0, le=1, description="Confidence score between 0 and 1")
    named_entities: List[NamedEntity] = Field(default_factory=list)
    april_events: List[EventSummary] = Field(default_factory=list)