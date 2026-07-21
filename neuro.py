import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
import os

class DiabetesNN:
    def __init__(self, input_size, hidden_size=32, learning_rate=0.001, epochs=50, batch_size=16):
        """
        :param input_size: Размер входных данных (количество признаков)
        :param hidden_size: Размер скрытого слоя (по умолчанию 32)
        :param learning_rate: Скорость обучения (по умолчанию 0.001)
        :param epochs: Количество эпох (по умолчанию 50)
        :param batch_size: Размер батча (по умолчанию 16)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()  # Логистическая функция для расчета вероятности
        ).to(self.device)
        self.criterion = nn.BCELoss()  # Бинарная кросс-энтропия для классификации
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.epochs = epochs
        self.batch_size = batch_size

    def load_data(self, file_path):
        """Загружает данные из CSV-файла и делит на тренировочную, валидационную и тестовую выборки."""
        data = pd.read_csv(file_path)
        X = data.iloc[:, :-1].values  # Все столбцы, кроме последнего (признаки)
        y = data.iloc[:, -1].values  # Последний столбец (целевая переменная)

        # Деление данных на тренировочную, валидационную и тестовую выборки (60%/20%/20%)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Конвертация в тензоры
        self.X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(self.device)
        self.X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        self.y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(self.device)
        self.X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        self.y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(self.device)

    def train(self):
        """Обучает модель."""
        for epoch in range(self.epochs):
            self.model.train()
            permutation = torch.randperm(self.X_train.size(0))

            for i in range(0, self.X_train.size(0), self.batch_size):
                indices = permutation[i:i + self.batch_size]
                batch_X, batch_y = self.X_train[indices], self.y_train[indices]

                # Прямой проход
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

                # Обратный проход и оптимизация
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Валидация
            val_loss = self.validate()
            print(f"Epoch {epoch + 1}/{self.epochs}, Validation Loss: {val_loss:.4f}")

    def validate(self):
        """Проверяет модель на валидационной выборке."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.X_val)
            loss = self.criterion(outputs, self.y_val)
        return loss.item()

    def test(self):
        """Оценивает качество модели на тестовой выборке."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.X_test)
            predictions = (outputs > 0.5).float()
            accuracy = (predictions == self.y_test).float().mean()
        print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")

    def save_model(self, save_path):
        """Сохраняет веса модели."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")


if __name__ == "__main__":
    # Параметры
    file_path = "database/data.csv"
    save_path = "training/model.pth"
    # Создаем объект нейронной сети
    input_size = 8  # Укажите количество признаков в данных
    model = DiabetesNN(input_size=input_size, hidden_size=64, learning_rate=0.001, epochs=2000, batch_size=192)

    # Загружаем данные и обучаем модель
    model.load_data(file_path)
    model.train()
    model.test()

    # Сохраняем веса
    model.save_model(save_path)
