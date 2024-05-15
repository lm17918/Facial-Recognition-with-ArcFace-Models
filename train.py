import hydra
import torch
import torch.nn.functional as F
import torch.optim as optim
import tqdm

from utils.network import ArcFaceLoss, ResNetArcFace, ResNetDreamArcFace
from utils.utils import dataset_setup


@hydra.main(version_base="1.1", config_path="config", config_name="config_train")
def main(cfg):
    # Initialize model
    model = ResNetArcFace(
        embedding_size=cfg.embedding_size, num_classes=cfg.num_classes
    )
    # model = ResNetDreamArcFace(
    #     embedding_size=cfg.embedding_size, num_classes=cfg.num_classes
    # )

    # Define optimizer and ArcFace loss criterion
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    criterion = ArcFaceLoss(
        embedding_size=cfg.embedding_size, num_classes=cfg.num_classes
    )

    train_loader = dataset_setup(cfg.data_dir)

    # Training loop
    for epoch in tqdm.tqdm(range(cfg.num_epochs)):
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
    torch.save(model.state_dict(), f"{cfg.save_path}/test.pth")


if __name__ == "__main__":
    main()
