# Flask is the web framework that turns your Python code into an API
# request lets you read what was sent to your API
# jsonify turns Python dictionaries into JSON responses
from flask import Flask, request, jsonify

# Our classifier from the models layer — knows nothing about Flask
from src.models.classifier import DocumentClassifier

# Our database functions — create the table and save results
from src.database.db import create_table, save_classification

# Create the Flask app — this is the actual web server
app = Flask(__name__)

# Create one classifier instance — reused for every request, not recreated each time
# Creating it once saves time and resources
classifier = DocumentClassifier()


# app.app_context() gives Flask the context it needs to run code at startup
# create_table() runs once when the app starts
# IF NOT EXISTS means it will never duplicate the table even if you restart the app
@app.before_request
def startup():
    create_table()


# @app.route means "when someone calls this URL, run this function"
# /health is the URL, GET means the caller is just fetching, not sending data
@app.route("/health", methods=["GET"])
def health():
    # Return a simple JSON response confirming the API is alive
    # Every production system has a health endpoint — used by load balancers and monitoring tools
    return jsonify({"status": "ok"})


# /classify is the main endpoint
# POST means the caller is sending data to us — the document to classify
@app.route("/classify", methods=["POST"])
def classify():
    # Read the JSON body from the incoming request
    data = request.get_json()

    # If nothing was sent, or the text key is missing, reject the request immediately
    # 400 means bad request — the caller did something wrong
    if not data or "text" not in data:
        return jsonify({"error": "Request body must contain a 'text' field"}), 400

    # Pull the text value out of the request body
    text = data["text"]

    # If text exists but is blank or just spaces, reject it
    # .strip() removes all whitespace before checking
    if not text.strip():
        return jsonify({"error": "Text field cannot be empty"}), 400

    # Send the text to the classifier and get a ClassificationResult back
    # This is the only line that calls OpenAI — everything else is our code
    result = classifier.classify(text=text)

    # Save the original request and the result permanently to PostgreSQL
    # This happens before we return the result — we never lose a record
    save_classification(input_text=text, result=result)

    # .model_dump() converts the Pydantic object into a plain Python dictionary
    # jsonify then turns that dictionary into a JSON response
    return jsonify(result.model_dump())


# /history lets you query the last 10 classifications from the database
# GET means no data is being sent — just fetching records
@app.route("/history", methods=["GET"])
def history():
    from src.database.db import get_connection
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, input_text, business, sports, entertainment, confidence, named_entities, april_events, created_at
        FROM classifications
        ORDER BY created_at DESC
        LIMIT 5
    """)
    rows = cursor.fetchall()
    # Get column names from cursor so order does not matter
    columns = [desc[0] for desc in cursor.description]
    cursor.close()
    conn.close()

    results = []
    for row in rows:
        # Zip column names with row values into a dictionary automatically
        row_dict = dict(zip(columns, row))
        # Trim input text to 100 characters for readability
        row_dict["input_text"] = row_dict["input_text"][:100]
        # Convert timestamp to string so jsonify can handle it
        row_dict["created_at"] = str(row_dict["created_at"])
        results.append(row_dict)

    return jsonify(results)


# This block only runs if you execute this file directly
# debug=True means Flask auto-reloads when you change code
# host="0.0.0.0" means accept connections from any machine, not just localhost
# port=5000 is the port the API listens on
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)