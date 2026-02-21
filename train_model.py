import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse

# Dependencias para Debugging y Profiling
from smdebug import pytorch as smd


def create_data_loaders(data_dir, batch_size):
    '''
    Crea data loaders para Train, Valid y Test.
    '''
    # Transformaciones con Resize fijo a (224, 224) para evitar errores de tamaño
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Cargar datasets
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
    valid_dataset = datasets.ImageFolder(os.path.join(data_dir, "valid"), transform=test_transform)
    test_dataset  = datasets.ImageFolder(os.path.join(data_dir, "test"),  transform=test_transform)

    # Crear loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size)

    return train_loader, valid_loader, test_loader, len(train_dataset.classes)

def test(model, loader, criterion, device, metric_name="validation"):
    '''
    Evalúa el modelo. metric_name ayuda a distinguir logs de valid vs test.
    '''
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    # SageMaker lee este print para el HPO Tuner
    print(f"{metric_name}:accuracy={accuracy}")
    return accuracy

def train(model, train_loader, criterion, optimizer, device, epochs):
    '''
    Entrena el modelo
    '''
    try:
        hook = smd.Hook.create_from_json_file()
    except:
        hook = None

    if hook:
        hook.register_module(model)
        hook.register_loss(criterion)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch} Loss: {running_loss / len(train_loader)}")
    
    return model

def net(num_classes):
    '''
    Inicializa ResNet50 con Fine-tuning en la última capa
    '''
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    
    for param in model.parameters():
        param.requires_grad = False

    # Descongelar capa 4 para mejor ajuste si es necesario
    for param in model.layer4.parameters():
        param.requires_grad = True

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    # SageMaker paths
    data_dir = os.environ["SM_CHANNEL_TRAIN"]
    model_dir = os.environ["SM_MODEL_DIR"]

    # Obtener los 3 loaders
    train_loader, valid_loader, test_loader, num_classes = create_data_loaders(data_dir, args.batch_size)

    model = net(num_classes)
    model.to(device)

    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # 1. Entrenar
    model = train(model, train_loader, loss_criterion, optimizer, device, args.epochs)
    
    # 2. Validar (Esto es lo que el Tuner observa)
    print("Testing on Validation Set...")
    test(model, valid_loader, loss_criterion, device, metric_name="validation")

    # 3. Test Final
    print("Testing on Test Set...")
    test(model, test_loader, loss_criterion, device, metric_name="test")

    # Guardar modelo
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    main(args)