import torch
import torch.nn as nn
import torchvision.transforms as transforms
import json
import io
from PIL import Image

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64*8*8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def model_fn(model_dir):
    model = SimpleCNN()
    model.load_state_dict(torch.load(f"{model_dir}/model.pth"))
    model.eval()
    return model

def input_fn(request_body, request_content_type):
    data = json.loads(request_body)
    tensor = torch.tensor(data).float()
    return tensor

def predict_fn(input_data, model):
    with torch.no_grad():
        output = model(input_data)
        return output.argmax(dim=1).tolist()

def output_fn(prediction, content_type):
    return json.dumps(prediction)
