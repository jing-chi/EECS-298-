from train import train, test
import torch

if __name__ == '__main__':
    train_losses, test_losses, model = train()
    test_predictions, y_test_actual = test()

    from matplotlib import pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.title('Training and Testing Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.show()

    plt.figure(figsize=(15, 6))
    plt.plot(y_test_actual, label='Actual Temperature', alpha=0.7)
    plt.plot(test_predictions, label='Predicted Temperature', alpha=0.7)
    plt.title('Actual vs Predicted Temperature')
    plt.xlabel('Time Steps')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.show()

    torch.save(model.state_dict(), 'weather_lstm_model.pth')