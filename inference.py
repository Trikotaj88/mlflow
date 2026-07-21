import torch
import torch.nn as nn

class DiabetesNN:
    def __init__(self, input_size, hidden_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        ).to(self.device)

    def load_model(self, model_path):
        """Загружает веса модели из файла."""
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {model_path}")

    def predict(self, input_data):
        """Делает предсказание для входных данных."""
        input_tensor = torch.tensor(input_data, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
        return output.cpu().numpy()

if __name__ == "__main__":
    # Загрузка модели
    model_path = "training/model.pth"
    input_size = 8  # Количество признаков
    model = DiabetesNN(input_size=input_size, hidden_size=64)
    model.load_model(model_path)

    # Пример предсказания
    sample_input = [[6.0,148.0,72.0,35.0,0.0,33.6,0.627,50], [8,183,64,0,0,23.3,0.672,32]]  # Пример входных данных
    prediction = model.predict(sample_input)
    print(f"Prediction: {prediction}")
