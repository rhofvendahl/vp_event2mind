from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from event2mind_hack import load_event2mind_archive
from allennlp.predictors.predictor import Predictor

app = Flask(__name__)
CORS(app)

archive = load_event2mind_archive('data/event2mind.tar.gz')
event2mind_predictor = Predictor.from_archive(archive)

def predict(source):
    return event2mind_predictor.predict(source=source)

@app.route('/predict', methods=['POST'])
def predict():
    content = request.get_json()
    predictions = [event2mind_predictor.predict(source=source) for source in content['sources']]
    return jsonify({
        'predictions': predictions
    })

if __name__ == "__main__":
    app.run()
