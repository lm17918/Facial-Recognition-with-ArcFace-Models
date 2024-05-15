import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ArcFaceLoss(nn.Module):
    def __init__(self, embedding_size, num_classes, margin=0.5, scale=64.0):
        super(ArcFaceLoss, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, targets):
        normalized_embeddings = F.normalize(embeddings)
        cosine_sim = self.cosine_similarity(
            normalized_embeddings, F.normalize(self.weight)
        )
        theta = torch.acos(torch.clamp(cosine_sim, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = F.one_hot(targets, self.num_classes).float()

        logits = self.scale * torch.cos(theta + self.margin * (one_hot - cosine_sim))
        return logits


class ResNetArcFace(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super(ResNetArcFace, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embedding_size)
        self.arcface = ArcFaceLoss(embedding_size, num_classes)

    def forward(self, x):
        return self.resnet(x)


class DREAMModule(nn.Module):
    def __init__(
        self, embedding_dim, num_angles=20
    ):  # Increased num_angles for more granular coverage
        super(DREAMModule, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_angles = num_angles
        self.fc1 = nn.Linear(embedding_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)

        # Learnable weights for each angle
        self.angle_weights = nn.Parameter(torch.randn(num_angles, embedding_dim))

    def forward(self, x):
        # x: [batch_size, embedding_dim]

        # Expand angle weights to match the batch size
        angles = self.angle_weights.unsqueeze(0)  # [1, num_angles, embedding_dim]
        angles = angles.repeat(
            x.size(0), 1, 1
        )  # [batch_size, num_angles, embedding_dim]

        # Expand input features to match the number of angles
        x_expanded = x.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        x_expanded = x_expanded.repeat(
            1, self.num_angles, 1
        )  # [batch_size, num_angles, embedding_dim]

        # Calculate attention scores based on the similarity between input features and angle weights
        attention_scores = F.softmax(
            torch.sum(x_expanded * angles, dim=2), dim=1
        )  # [batch_size, num_angles]

        # Compute weighted sum of angle weights
        weighted_angles = torch.sum(
            attention_scores.unsqueeze(2) * angles, dim=1
        )  # [batch_size, embedding_dim]

        # Adjust embeddings with the computed weighted angles
        x = F.relu(self.fc1(x + weighted_angles))
        x = self.fc2(x)
        return x


class ResNetWithDREAM(nn.Module):
    def __init__(
        self, base_model, embedding_dim, num_angles=20
    ):  # Match num_angles with DREAM module
        super(ResNetWithDREAM, self).__init__()
        self.base_model = base_model
        # self.base_model.fc = nn.Identity()  # Remove the original fully connected layer
        self.dream = DREAMModule(embedding_dim, num_angles)

    def forward(self, x):
        x = self.base_model(x)  # Extract features using ResNet
        x = self.dream(x)  # Apply DREAM module
        return x


class ResNetDreamArcFace(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super(ResNetDreamArcFace, self).__init__()
        resnet = models.resnet18(pretrained=True)
        # embedding_dim = self.model.fc.in_features  # Typically 2048 for ResNet-50
        resnet.fc = nn.Linear(resnet.fc.in_features, embedding_size)
        # Create the model with DREAM
        self.model = ResNetWithDREAM(resnet, embedding_size)

        # resnet.fc = nn.Linear(resnet.fc.in_features, embedding_size)

        self.arcface = ArcFaceLoss(embedding_size, num_classes)

    def forward(self, x):
        return self.model(x)
