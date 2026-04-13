import instructor
from openai import OpenAI
from src.schemas.classification import ClassificationResult
from src.config.settings import settings

SYSTEM_PROMPT = """
You are an AI assistant designed to analyze text data and classify it according to specific categories related to business, sports, and entertainment.
Your role is to extract structured information that helps categorize the content quickly and accurately.

Business Context:
- Texts may include news, reports, or summaries involving various subcategories within business, sports, and entertainment.
- You will identify key named entities such as media personalities and their roles or jobs, which may not be limited to a fixed predefined list.
- Your classification must include confidence scores to indicate reliability.

Your tasks for each input text are to:
1. Assign the most relevant categories in business, sports, and entertainment (if applicable).
2. Extract named entities with their full names and job descriptions (job can be free text, not limited to a fixed list).
3. APRIL EVENTS - HIGH RECALL: Extract ANY events that could occur in April (any year before 2006). Include events with explicit April mentions OR spring/seasonal references that could indicate April timing. When uncertain about April timing, INCLUDE with lower confidence rather than exclude.
4. Provide a confidence score between 0 and 1 for your classification accuracy.

Important notes:
- Use free-text for the job type of named entities, allowing for flexibility in new or uncommon job titles.
- Base your analysis solely on the provided text without assumptions.
- For April events: Maximum recall approach - if there is ANY possibility it could be April, include it.
- If unsure, lower the confidence score accordingly.
- The output should be structured as JSON with keys: business, sports, entertainment, confidence, named_entities, april_events.
"""


class DocumentClassifier:

    def __init__(self):
        openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.client = instructor.patch(openai_client)
        self.model = settings.DEFAULT_MODEL

    def classify(self, text: str) -> ClassificationResult:
        response = self.client.chat.completions.create(
            model=self.model,
            response_model=ClassificationResult,
            temperature=settings.TEMPERATURE,
            max_retries=settings.MAX_RETRIES,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text}
            ]
        )
        return response