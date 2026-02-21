import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys
import os

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = "model.pth"
TRAIN_DIR = "dogImages/train"  # solo para recuperar class names
IMAGE_PATH = sys.argv[1]       # imagen a evaluar
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------  dogImages/valid/001.Affenpinscher/Affenpinscher_00038.jpg
# TRANSFORM (igual al testing_transform)
# -------------------------
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------
# LOAD CLASS NAMES
# -------------------------
from torchvision.datasets import ImageFolder
train_dataset = ImageFolder(root=TRAIN_DIR)
class_names = train_dataset.classes
num_classes = len(class_names)

# -------------------------
# LOAD MODEL
# -------------------------
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -------------------------
# LOAD IMAGE
# -------------------------
image = Image.open(IMAGE_PATH).convert("RGB")
image = transform(image)
image = image.unsqueeze(0).to(DEVICE)

# -------------------------
# INFERENCE
# -------------------------
with torch.no_grad():
    outputs = model(image)
    probabilities = torch.softmax(outputs, dim=1)
    top_prob, top_class = probabilities.topk(5, dim=1)

# -------------------------
# PRINT RESULTS
# -------------------------
print("\nTop 5 Predictions:\n")
for i in range(5):
    class_name = class_names[top_class[0][i].item()]
    prob = top_prob[0][i].item()
    print(f"{i+1}. {class_name} - {prob:.4f}")

print("\nPredicted class:", class_names[top_class[0][0].item()])
