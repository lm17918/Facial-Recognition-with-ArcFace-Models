import hydra
import torch
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import matplotlib.pyplot as plt
import os

from utils.network import ArcFaceLoss, ResNetArcFace, ResNetDreamArcFace
from utils.utils import dataset_setup


@hydra.main(version_base="1.1", config_path="config", config_name="config_train")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize models
    model = ResNetArcFace(embedding_size=cfg.embedding_size, num_classes=cfg.num_classes)
    model_with_dream = ResNetDreamArcFace(embedding_size=cfg.embedding_size, num_classes=cfg.num_classes)

    models = [model, model_with_dream]
    model_names = ["ResNetArcFace", "ResNetDreamArcFace"]
    losses = {name: [] for name in model_names}

    for model, model_name in zip(models, model_names):
        model.to(device)

        # Define optimizer and ArcFace loss criterion
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        criterion = ArcFaceLoss(embedding_size=cfg.embedding_size, num_classes=cfg.num_classes).to(device)

        train_loader = dataset_setup(cfg.data_dir, device=device)

        # Training loop
        for epoch in tqdm.tqdm(range(cfg.num_epochs)):
            model.train()
            running_loss = 0.0

            for images, labels in tqdm.tqdm(train_loader):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                embeddings = model(images)
                logits = criterion(embeddings, labels)  # Compute ArcFace logits
                loss = F.cross_entropy(logits, labels)  # Use cross-entropy loss with logits
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            losses[model_name].append(avg_loss)
            print(f"{model_name} - Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        # Save the trained model
        torch.save(model.state_dict(), f"{cfg.save_path}/{model_name}.pth")

    # Plot and save the training loss
    plt.figure()
    for model_name in model_names:
        plt.plot(losses[model_name], label=model_name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)

    # Save the plot as an image
    if not os.path.exists(cfg.save_path):
        os.makedirs(cfg.save_path)
    plt.savefig(os.path.join(cfg.save_path, "training_loss_comparison.png"))

if __name__ == "__main__":
    main()
