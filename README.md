# Structured Text Classification with Large Language Models (News-Structurizer)
## Overview

This project leverages large language models (LLMs) to classify BBC news articles into detailed subcategories within **Business**, **Sports**, and **Entertainment**. It also extracts named entities with their job titles and summarizes events that occur **strictly in April**. The output is returned as structured JSON for easy downstream integration.

### Key Features
- **Multi-label Classification**: Tags articles under business, sports, and entertainment subcategories.
- **Named Entity Extraction**: Identifies full names and **free-text** job titles.
- **Event Summarization**: Includes **only** April events. Ignores all others.
- **Confidence Scores**: Scored from 0 to 1 based on certainty of classification.

All tasks are executed using OpenAI's GPT models guided by a robust system prompt for consistent and precise output.

## Environment Setup

The environment is defined in `environment.yml`. To get started:

```bash
conda env create -f environment.yml
conda activate llms
```

### Create a .env file and add your OpenAI API key and launch jupyter lab


## Dataset Information

This project uses the [BBC News article dataset](http://mlg.ucd.ie/datasets/bbc.html), a benchmark collection for text classification research.

> D. Greene and P. Cunningham.  
> "Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering",  
> *Proceedings of ICML 2006*.  
> http://mlg.ucd.ie/datasets/bbc.html

**Note**: Only articles before 2006 are valid. Any event summaries referring to later years are automatically flagged as incorrect.

## Project Details

### Classification & Extraction Logic

- **Categories**: Uses Python `enum` to enforce consistent subcategories for Business, Sports, and Entertainment.
- **Named Entities**: Extracts full names and **free-text job descriptions**.
- **April-Only Events**: Only events in **April** are included; all others are discarded.
- **Confidence Scores**: Scores between `0` and `1` reflect classification certainty. Lower scores mean uncertainty.

## System Prompts (Iterations)

**Prompt 1**: Basic instructions for categorization and extraction.  
**Prompt 2**: Added confidence scoring and April-only constraint.  
**Prompt 3** *(Final & Best)*: Included:
- Free-text job descriptions
- Strict April filtering
- Confidence adjustment
- Strict JSON output structure

The third prompt gave the **best results** and was used for final evaluation.

## Final System Prompt
You are an AI assistant designed to analyze text data and classify it according to specific categories related to business, sports, and entertainment.
Your role is to extract structured information that helps categorize the content quickly and accurately.

Business Context:

- Texts may include news, reports, or summaries involving various subcategories within business, sports, and entertainment.
- You will identify key named entities such as media personalities and their roles or jobs, which may not be limited to a fixed predefined list.
- Your classification must include confidence scores to indicate reliability.

Your tasks for each input text are to:
- Assign the most relevant categories in business, sports, and entertainment (if applicable).
- Extract named entities with their full names and job descriptions (job can be free text, not limited to a fixed list).
- Summarize any events mentioned strictly for those occurring in April. Don't return any other month. If you are not sure, do not return false info.
- Provide a confidence score between 0 and 1 for your classification accuracy.

Important notes:
- Use free-text for the job type of named entities, allowing for flexibility in new or uncommon job titles.
- Base your analysis solely on the provided text without assumptions.
- If unsure, lower the confidence score accordingly.
- The output should be structured and formatted as JSON matching the following keys:
    business
    sports
    entertainment
    confidence
    named_entities (each with name and job)
    april_events (list of event summaries with date, description)
    Any year above 2006 is false because the information is before 2006.

## Model Evaluation & Cost Comparison

| Model         | Accuracy | Precision | Recall  | F1 Score | Approx. Cost per Run  |
|---------------|----------|-----------|---------|----------|-----------------------|
| GPT-3.5 Turbo | 0.5409   | 0.0980    | 0.9722  | 0.1781   | ~$0.80                |
| GPT-4o-mini   | 0.8770   | 0.2851    | 0.9306  | 0.4365   | ~$0.40                |
| GPT-4o        | 0.8827   | 0.3038    | 1.0000  | 0.4660   | ~$3.00                |

- **GPT-4o-mini** is the most cost-effective.
- **GPT-4o** had perfect recall and captured **all April events**.
- All models performed strongly in named entity and job title extraction.
- Average classification confidence: **~90%** across all models.

## Code Structure

- **bbc.py**: Modularized logic for classification, extraction, JSON formatting, and evaluation.
- **Enums**: Used to enforce strict subcategories for consistency.
- **Pydantic Models**: Validate output structure (e.g., confidence, events, named entities).
- **Run_BBC_Process.ipynb**: Interactive notebook that executes the complete classification and extraction pipeline.

## Usage Instructions

1. Open `Run_BBC_Process.ipynb` in JupyterLab.
2. Restart the kernel to reset state.
3. Run all cells in sequence to:
   - Load and clean BBC dataset.
   - Classify and extract using OpenAI models.
   - Save structured results to disk.
   - Evaluate performance metrics.

All core logic is handled inside `bbc.py`.

## Notes

- **Named entity jobs** are free-text to handle uncommon or emerging roles.
- **Only April events** are included to prevent irrelevant or false data.
- **Confidence scores** help rank results and support downstream filtering or manual review.
- Cost estimates are based on current OpenAI pricing.


