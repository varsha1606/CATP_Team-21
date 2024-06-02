from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required, current_user
from predictor.load_model import load_model
from predictor.predict import predict_next_word

views = Blueprint('views', __name__)

# Model and dictionaries paths
model_path = "lstm_model.pth"
model, word_to_idx, idx_to_word = load_model(model_path)

if model is None:
    raise ValueError("Failed to load the model. Please check the model path and ensure the model file exists and is accessible.")

@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    return render_template("home.html", user=current_user)

@views.route('/predict', methods=['POST'])
def predict():
    input_sentence = request.json.get("input_sentence")
    predicted_word = predict_next_word(model, word_to_idx, idx_to_word, input_sentence)
    return jsonify({"predicted_word": predicted_word})
