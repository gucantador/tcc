import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# https://www.youtube.com/watch?v=pDdP0TFzsoQ&t=45s

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 4
batch_size = 4
learning_rate = 0.001

# ============================================================
# TRANSFORM PARA O CIFAR-10
# ============================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
])
# ============================================================
# DATASETS
# ============================================================
train_dataset = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
# ============================================================
# DATA LOADERS
# ============================================================

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=0)

# ============================================================
# CLASSES
# ============================================================

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# ============================================================
# MODELO ORIGINAL
# ============================================================

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x

# ============================================================
# TREINAMENTO
# ============================================================

def train_net(_model,_device,_train_loader,_optimizer,save_path):

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
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}")

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

        n_correct_class = [0 for _ in range(10)]
        n_samples_class = [0 for _ in range(10)]

        for images, labels in _test_loader:
            images = images.to(_device)
            labels = labels.to(_device)

            outputs = _model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            n_samples += labels.size(0)

            n_correct += (
                predicted == labels
            ).sum().item()

            for i in range(labels.size(0)):

                label = labels[i]
                pred = predicted[i]

                if label == pred:

                    n_correct_class[
                        label
                    ] += 1

                n_samples_class[
                    label
                ] += 1

        acc = (
            100.0 *
            n_correct /
            n_samples
        )

        print(
            f"Test Accuracy: {acc}%"
        )

        for i in range(10):

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

    optimizer = torch.optim.Adam(
        _model.parameters(),
        lr=learning_rate
    )

    train_net(
        _model,
        _device,
        _train_loader,
        optimizer,
        save_path
    )

# ============================================================
# CARREGAR NET
# ============================================================

def get_net(need_evaluate):
    net = Net().to(device)
    error = False
    saved_path = 'models/trained_net_base_model.pth'

    try:
        net.load_state_dict(torch.load(saved_path, map_location=device ) )

        print("Loaded model")

    except FileNotFoundError:

        error = True

        train_net_(
            net,
            device,
            train_loader,
            saved_path
        )

    finally:

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
# TRANSFORM PARA IMAGENS DA INTERNET
# ============================================================

internet_transform = transforms.Compose([
    # Mantém a proporção da imagem
    transforms.Resize(40),
    # Pega uma região central 32x32
    transforms.CenterCrop(32),
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

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = internet_transform(image)
    # Adiciona dimensão do batch
    image = image.unsqueeze(0)
    return image

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # CARREGAR MODELO
    # ========================================================

    model = get_net(False)

    model.eval()

    # ========================================================
    # IMAGENS PARA TESTAR
    # ========================================================

    images_paths = [
        r'C:\Users\Gustavo\Desktop\codes\tcc\imgs\image1.png',
        r'C:\Users\Gustavo\Desktop\codes\tcc\imgs\image2.jpg',
        r'C:\Users\Gustavo\Desktop\codes\tcc\imgs\dog.jpg',
        r'C:\Users\Gustavo\Desktop\codes\tcc\imgs\car.jpg'
    ]

    images = [
        load_image(img)
        for img in images_paths
    ]

    # ========================================================
    # CLASSIFICAR IMAGENS
    # ========================================================

    with torch.no_grad():
        for image in images:
            # Importante:
            # modelo e imagem precisam estar no mesmo device
            image = image.to(device)
            output = model(image)
            _, predicted = torch.max(output,1)

            predicted_class = (classes[predicted.item()])

            print(f"Predicted: {predicted_class}")