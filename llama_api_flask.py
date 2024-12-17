from flask import Flask, request, jsonify
from flask_restx import Api, Resource, reqparse
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

# Flask app initialization
app = Flask(__name__)
api = Api(app)

# Namespace setup
ns = api.namespace("ai", description="Simple AI Query API")

# Ollama model configuration
LLAMA_MODEL_URL = "http://127.0.0.1:11434"
MODEL_NAME = "llama3.2"
Settings.llm = Ollama(model=MODEL_NAME, base_url=LLAMA_MODEL_URL, request_timeout=120.0)

# Request parser for query
parser = reqparse.RequestParser()
parser.add_argument("query", type=str, required=True, help="Query for the AI model")


@ns.route("/predict")
class Predict(Resource):
    @ns.expect(parser)
    def get(self):
        """
        Handle GET request with a query parameter.
        """
        try:
            # Parse query from the URL
            args = parser.parse_args()
            ai_query = args.get("query")

            if not ai_query:
                return {"error": "No query provided"}, 400

            # Send query to Ollama using the `complete` method
            llama_model = Settings.llm
            response = llama_model.complete(prompt=ai_query)

            # Ensure the response is JSON serializable
            if hasattr(response, "to_dict"):
                response_data = response.to_dict()  # Convert to dictionary if available
            else:
                response_data = {"text": str(response)}

            # Return the serialized response
            return {"response": response_data}, 200
        except Exception as e:
            return {"error": str(e)}, 500


# Add namespace to API
api.add_namespace(ns, path="/api")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
