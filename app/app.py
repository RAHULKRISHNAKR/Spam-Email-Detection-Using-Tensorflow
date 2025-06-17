from flask import Flask, render_template, request, jsonify
import os
import sys

# Add the project root directory to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import the SpamPredictor from the src directory instead
from src.app.predict import SpamPredictor

app = Flask(__name__)

# Initialize the SpamPredictor
MODEL_PATH = os.path.join(project_root, 'output', 'spam_classifier_model.h5')
TOKENIZER_PATH = os.path.join(project_root, 'output', 'tokenizer', 'tokenizer.pickle')

# Initialize the predictor only when needed (lazy loading)
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        predictor = SpamPredictor(
            model_path=MODEL_PATH,
            tokenizer_path=TOKENIZER_PATH,
            threshold=0.3  # Add this line to lower the threshold
        )
    return predictor

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make a prediction based on the input text"""
    data = request.form
    text = data.get('message', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Get predictor and make prediction
    try:
        predictor = get_predictor()
        result = predictor.predict(text)
        
        # Get the first prediction (since we're only sending one text)
        prediction = result['predictions'][0]
        
        return render_template(
            'result.html', 
            prediction=prediction,
            text=text
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for making predictions"""
    data = request.get_json()
    text = data.get('message', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Get predictor and make prediction
    try:
        predictor = get_predictor()
        result = predictor.predict(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)