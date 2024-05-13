import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
from utils.network import ResNetArcFace, ArcFaceLoss
from torchvision import datasets, transforms
import tqdm
from utils.network import ResNetArcFace, ArcFaceLoss
from utils.utils import dataset_setup

num_classes = 307


# Prepare dataset and DataLoader
# Assuming train_loader is set up correctly

# Initialize model
model = ResNetArcFace(embedding_size=128, num_classes=num_classes)

# Define optimizer and ArcFace loss criterion
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
criterion = ArcFaceLoss(embedding_size=128, num_classes=num_classes)


data_dir = "./data/identity_dataset/train"
train_loader = dataset_setup(data_dir)

# Training loop
num_epochs = 10
for epoch in tqdm.tqdm(range(num_epochs)):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm.tqdm(train_loader):
        optimizer.zero_grad()
        embeddings = model(images)
        logits = criterion(embeddings, labels)  # Compute ArcFace logits
        loss = F.cross_entropy(logits, labels)  # Use cross-entropy loss with logits
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# Save the trained model
torch.save(model.state_dict(), "resnet_arcface.pth")
