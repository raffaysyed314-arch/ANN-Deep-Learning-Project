import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import sys

# Ensure Python can find our custom modules
sys.path.append('.')
from models.pretrained_cnns import get_resnet18, get_mobilenet_v2, get_efficientnet_b0
from scripts.create_dataloaders import prepare_data_and_loaders

def train_pretrained_model(model, train_loader, val_loader, model_name, epochs=10):
    print(f"\n--- Fine-Tuning {model_name} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # CRITICAL: Only optimize the layers that are unfrozen (the new classifier heads)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
    os.makedirs('saved_models', exist_ok=True)
    torch.save(model.state_dict(), f'saved_models/{model_name}.pth')
    plot_curves(train_losses, val_losses, train_accs, val_accs, model_name)

def plot_curves(t_loss, v_loss, t_acc, v_acc, name):
    os.makedirs('results/plots', exist_ok=True)
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(t_loss, label='Train Loss')
    plt.plot(v_loss, label='Val Loss')
    plt.title(f'{name} - Loss Curve')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(t_acc, label='Train Acc')
    plt.plot(v_acc, label='Val Acc')
    plt.title(f'{name} - Accuracy Curve')
    plt.legend()
    
    plt.savefig(f'results/plots/{name}_curves.png')
    plt.close()
    print(f"Saved learning curves to results/plots/{name}_curves.png")

if __name__ == "__main__":
    train_loader, val_loader, _ = prepare_data_and_loaders()
    if train_loader:
        models = [
            (get_resnet18(), "ResNet18"),
            (get_mobilenet_v2(), "MobileNetV2"),
            (get_efficientnet_b0(), "EfficientNetB0")
        ]
        for model, name in models:
            train_pretrained_model(model, train_loader, val_loader, name, epochs=10)
