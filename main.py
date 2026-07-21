from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import os

app = Flask(__name__)


class DiabetesNN:
    def __init__(self, input_size, hidden_size=32):
        """
        Инициализирует модель нейронной сети.
        :param input_size: Размер входных данных (количество признаков).
        :param hidden_size: Размер скрытого слоя (по умолчанию 32).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        ).to(self.device)

    def load_model(self, model_path):
        """Загружает веса модели."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена по пути: {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict(self, input_data):
        """Предсказывает вероятность диабета для входных данных."""
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor).item()
        return output * 100  # Возвращает вероятность в процентах


# Инициализация модели
input_size = 8  # Количество признаков в данных
model_path = "training/model.pth"
diabetes_model = DiabetesNN(input_size=input_size, hidden_size=64)
diabetes_model.load_model(model_path)


@app.route("/predict", methods=["POST"])
def predict():
    """Эндпоинт для предсказания вероятности диабета."""
    try:
        data = request.json
        if not data or "features" not in data:
            return jsonify({"error": "Отсутствуют входные данные (features)."}), 400

        features = data["features"]
        if len(features) != input_size:
            return jsonify({"error": f"Ожидается {input_size} признаков, получено {len(features)}."}), 400

        # Предсказание модели
        probability = diabetes_model.predict(features)
        return jsonify({"probability": probability})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
