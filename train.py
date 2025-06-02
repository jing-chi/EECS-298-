import torch
from model import LSTM
from data  import data_process, create_dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
import torch.nn as nn
import numpy as np


features, target, dataset = data_process()
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(dataset)
X,y = create_dataset(scaled_data)

imput_size = len(features)
hidden_size = 128
num_layers = 2
output_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=LSTM(inputsize=imput_size, hiddensize=hidden_size, num_layers=num_layers, outputsize=output_size).to(device)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
train_losses = []
test_losses = []

def train():
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                test_loss += criterion(outputs, batch_y).item() * batch_x.size(0)

        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}')
    return train_losses, test_losses, model


def test():
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test.to(device)).cpu().numpy()

        temp_array = np.zeros((len(y_test), len(features) + 1))
        temp_array[:, -1] = test_predictions.flatten()

        test_predictions = scaler.inverse_transform(temp_array)[:, -1]

        temp_array[:, -1] = y_test.flatten()
        y_test_actual = scaler.inverse_transform(temp_array)[:, -1]

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(y_test_actual, test_predictions)
    rmse = np.sqrt(mean_squared_error(y_test_actual, test_predictions))
    r2 = r2_score(y_test_actual, test_predictions)

    print(f'Test MAE: {mae:.2f}')
    print(f'Test RMSE: {rmse:.2f}')
    print(f'Test R²: {r2:.4f}')
    return test_predictions, y_test_actual