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
