import os
import psycopg2
import json
from datetime import datetime


def get_connection():
    # Connect to PostgreSQL using environment variables
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "db"),
        port=os.getenv("POSTGRES_PORT", 5432),
        database=os.getenv("POSTGRES_DB", "bbc"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres")
    )


def create_table():
    # Create the classifications table if it does not exist
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS classifications (
            id SERIAL PRIMARY KEY,
            input_text TEXT NOT NULL,
            business VARCHAR,
            sports VARCHAR,
            entertainment VARCHAR,
            confidence FLOAT,
            named_entities JSONB,
            april_events JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()


def save_classification(input_text: str, result) -> None:
    # Save one classification result to the database
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO classifications 
        (input_text, business, sports, entertainment, confidence, named_entities, april_events)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """,
    (
        input_text,
        result.business.value if result.business else None,
        result.sports.value if result.sports else None,
        result.entertainment.value if result.entertainment else None,
        result.confidence,
        json.dumps([e.model_dump() for e in result.named_entities]),
        json.dumps([e.model_dump() for e in result.april_events])
    ))
    conn.commit()
    cursor.close()
    conn.close()