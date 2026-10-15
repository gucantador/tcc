import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import ela_split_save
# https://www.youtube.com/watch?v=pDdP0TFzsoQ&t=45s

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_ela = False

num_epochs = 4
batch_size = 64
learning_rate = 0.001

# ============================================================
# TRANSFORM PARA O CIFAR-10
# ============================================================
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])
# ============================================================
# DATASETS
# ============================================================
# train_dataset = torchvision.datasets.CIFAR10(
#     root='./data',
#     train=True,
#     download=True,
#     transform=transform
# )
# test_dataset = torchvision.datasets.CIFAR10(
#     root='./data',
#     train=False,
#     download=True,
#     transform=transform
# )
train_dataset = torchvision.datasets.ImageFolder(
    root=f'{'ela_results'if use_ela else 'original'}/train',
    transform=transform
)
test_dataset = torchvision.datasets.ImageFolder(
    root=f'{'ela_results'if use_ela else 'original'}/test',
    transform=transform
)
# ============================================================
# DATA LOADERS
# ============================================================

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)
train_loader_32 = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)
test_loader_32 = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

# ============================================================
# CLASSES
# ============================================================

classes = ("fake", "real")

# ============================================================
# MODELO ORIGINAL
# ============================================================

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Para entrada 512x512:
        # 512 -> 508 -> 254 -> 250 -> 125
        self.fc1 = nn.Linear(16 * 125 * 125, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # x = x.view(-1, 16 * 5 * 5)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x


# ============================================================
# SEGUNDO MODELO
# ============================================================

class NeuralNet(nn.Module):

    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 24, 5)

        self.fc1 = nn.Linear(24 * 5 * 5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 24 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ============================================================
# TREINAMENTO
# ============================================================

def train_net(
    _model,
    _device,
    _train_loader,
    _optimizer,
    save_path
):

    n_total_steps = len(_train_loader)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(_train_loader):
            images = images.to(_device)
            labels = labels.to(_device)

            outputs = _model(images)

            loss = criterion(
                outputs,
                labels
            )

            _optimizer.zero_grad()
            loss.backward()
            _optimizer.step()

            if (i + 1) % 2000 == 0:

                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Step [{i + 1}/{n_total_steps}], "
                    f"Loss: {loss.item():.4f}"
                )

    print("Finished Training")

    torch.save(_model.state_dict(),save_path)


# ============================================================
# AVALIAÇÃO DO MODELO ORIGINAL
# ============================================================

def evaluate(
    _model,
    _test_loader,
    _device
):
    _model.eval()
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_correct_class = [0 for _ in range(2)]

        n_samples_class = [0  for _ in range(2)]

        for images, labels in _test_loader:
            images = images.to(_device)
            labels = labels.to(_device)
            outputs = _model(images)

            _, predicted = torch.max(outputs,1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            for i in range(labels.size(0)):
                label = labels[i]
                pred = predicted[i]
                if label == pred:
                    n_correct_class[label] += 1
                n_samples_class[label] += 1

        acc = (100.0 *n_correct /n_samples)

        print(
            f"Test Accuracy: {acc}%"
        )

        for i in range(2):

            acc = (
                100.0 *
                n_correct_class[i] /
                n_samples_class[i]
            )

            print(
                f"Accuracy of {classes[i]}: {acc}%"
            )


# ============================================================
# TREINAR NET
# ============================================================

def train_net_(
    _model,
    _device,
    _train_loader,
    save_path
):
    try:
        optimizer = torch.optim.Adam( _model.parameters(),lr=learning_rate)
        train_net( _model,_device,_train_loader, optimizer,save_path)
        return True
    except Exception as e:
        print(e)
        return False

# ============================================================
# TREINAR NEURAL NET
# ============================================================

def train_neural_net(
    _model,
    _device,
    _train_loader,
    save_path
):

    neural_net = NeuralNet().to(_device)

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        neural_net.parameters(),
        lr=learning_rate,
        momentum=0.9
    )

    for epoch in range(30):

        running_loss = 0.0

        for i, (inputs, labels) in enumerate(
            _train_loader
        ):

            inputs = inputs.to(_device)
            labels = labels.to(_device)

            optimizer.zero_grad()

            outputs = neural_net(inputs)

            loss = loss_function(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        print(
            f'Loss: '
            f'{running_loss / len(_train_loader):.4f}'
        )

    torch.save(
        neural_net.state_dict(),
        save_path
    )


# ============================================================
# CARREGAR NET
# ============================================================

def get_net(need_evaluate):
    net = Net().to(device)
    error = False
    saved_path = 'models/trained_ela_net_base_model.pth' if use_ela else 'models/trained_net_base_model.pth'
    try:
        net.load_state_dict(torch.load(saved_path,map_location=device))
        print("Loaded model")
        success = True
    except FileNotFoundError:
        error = True
        success = train_net_(
            net,
            device,
            train_loader,
            saved_path
        )
    if not success:
        return None
    if error:
        net.load_state_dict(
            torch.load(
                saved_path,
                map_location=device
            )
        )

    if need_evaluate:
        evaluate(
            net,
            test_loader,
            device
        )

    return net


# ============================================================
# CARREGAR NEURAL NET
# ============================================================

def get_neural_net(need_evaluate):

    neural_net = NeuralNet().to(device)

    error = False

    saved_path = (
        'models/trained_neural_net_base_model.pth'
    )

    try:

        neural_net.load_state_dict(
            torch.load(
                saved_path,
                map_location=device
            )
        )

        print("Loaded model")

    except FileNotFoundError:

        error = True

        train_neural_net(
            neural_net,
            device,
            train_loader_32,
            saved_path
        )

    finally:

        if error:

            neural_net.load_state_dict(
                torch.load(
                    saved_path,
                    map_location=device
                )
            )

    if need_evaluate:

        evaluate_neural_net(
            neural_net,
            device,
            test_loader_32
        )

    return neural_net


# ============================================================
# AVALIAR NEURAL NET
# ============================================================

def evaluate_neural_net(
    _model,
    _device,
    _test_loader
):
    correct = 0
    total = 0
    _model.eval()
    with torch.no_grad():
        for images, labels in _test_loader:
            images = images.to(_device)
            labels = labels.to(_device)
            _output = _model(images)
            _, _predicted = torch.max(_output,1)
            total += labels.size(0)
            correct += (_predicted == labels).sum().item()
    accuracy = (100.0 * correct / total)

    print(f"Test Accuracy: {accuracy}%")


# ============================================================
# TRANSFORM PARA IMAGENS DA INTERNET
# ============================================================

internet_transform = transforms.Compose([
    # Mantém a proporção da imagem
    transforms.Resize((512, 512)),
    # Pega uma região central 32x32
    # transforms.CenterCrop(32),
    # Converte para Tensor
    transforms.ToTensor(),
    # Mesma normalização utilizada no treinamento
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


# ============================================================
# CARREGAR IMAGEM DA INTERNET
# ============================================================

def load_image(_image_path):
    # Converte o numpy array para PIL Image
    if use_ela:
        # Calcula a ELA da imagem original
        ela = ela_split_save.compute_ela(_image_path)
        # compute_ela retorna BGR (OpenCV).
        # O PIL espera RGB.
        ela = cv2.cvtColor(ela, cv2.COLOR_BGR2RGB)
        _image = Image.fromarray(ela).convert("RGB")
    else:
        _image = Image.open(_image_path).convert("RGB")
    _image = internet_transform(_image)
    # Adiciona dimensão do batch
    _image = _image.unsqueeze(0)
    return _image


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    use_ela = True
    # ========================================================
    # CARREGAR MODELO
    # ========================================================

    model = get_net(True)
    if model is None:
        exit()
    model.eval()

    # ========================================================
    # IMAGENS PARA TESTAR
    # ========================================================

    images_paths = [
        'original/validate/fake/4455_ela.png',
        'original/validate/fake/4456_ela.png',
        'original/validate/fake/4457_ela.png',
        'original/validate/fake/4458_ela.png',
        'original/validate/fake/4459_ela.png',

        'original/validate/real/4960_ela.png',
        'original/validate/real/4961_ela.png',
        'original/validate/real/4962_ela.png',
        'original/validate/real/4963_ela.png',

        'gustavo_ia.jpg',
        'gustavo_ia_gemini.jpg',
        'gustavo_ia_gemini_zap.jpg',
        'gustavo_real.jpg',
        'gustavo_zap.jpg',
        'images.jpg',
        'images (1).jpg',
    ]

    images_dict = {}

    for image_path in images_paths:
        images_dict[image_path] = load_image(image_path)

    # ========================================================
    # CLASSIFICAR IMAGENS
    # ========================================================

    with torch.no_grad():
        for image in images_dict:
            # Importante:
            # modelo e imagem precisam estar no mesmo device
            imageDevice = images_dict[image].to(device)
            output = model(imageDevice)
            _, predicted = torch.max(output,1)

            predicted_class = (classes[predicted.item()])

            print(f"Predicted: {predicted_class} - {image}")