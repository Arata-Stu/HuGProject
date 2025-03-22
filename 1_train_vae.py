import hydra
from omegaconf import DictConfig, OmegaConf
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from src.models.vae import CNN_VAE

class CheckpointManager:
    """
    指定された基本ディレクトリにチェックポイントを保存し、
    topk 個のベストな重みのみを保持する管理クラス。
    """
    def __init__(self, base_dir, topk):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.topk = topk
        self.checkpoints = []  # (loss, filepath) のリスト

    def update(self, epoch, loss, model):
        filename = f"cnn_vae_epoch{epoch}.pth"
        filepath = os.path.join(self.base_dir, filename)
        # チェックポイントの保存
        torch.save(model.state_dict(), filepath)
        print(f"Checkpoint saved: {filepath}")
        # チェックポイントリストに追加
        self.checkpoints.append((loss, filepath))
        # 損失の昇順（低いほうが良い）でソート
        self.checkpoints.sort(key=lambda x: x[0])
        # topk を超える場合は、最も損失が高い（リストの末尾）のチェックポイントを削除
        if len(self.checkpoints) > self.topk:
            worst_loss, worst_path = self.checkpoints.pop()
            if os.path.exists(worst_path):
                os.remove(worst_path)
                print(f"Removed old checkpoint: {worst_path}")

@hydra.main(config_path="config", config_name="train_vae")
def main(cfg: DictConfig):
    print("Loaded config:")
    print(OmegaConf.to_yaml(cfg))

    # 設定からパラメータを取得
    latent_dim = cfg.vae.latent_dim
    input_shape = tuple(cfg.input_shape)  # [C, H, W]
    cnn_cfg = OmegaConf.create({"ckpt_path": cfg.vae.cnn.ckpt_path})
    lr = cfg.lr
    batch_size = cfg.batch_size
    num_epochs = cfg.num_epochs

    # latent_dim と input_shape の情報を含むディレクトリ名を生成
    input_shape_str = "x".join(map(str, input_shape))
    checkpoint_dir = os.path.join(cfg.checkpoint.base_dir, f"vae_latent_{latent_dim}_shape_{input_shape_str}")
    log_dir = os.path.join(cfg.tensorboard.log_dir, f"vae_latent_{latent_dim}_shape_{input_shape_str}")

    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"TensorBoard log directory: {log_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # input_shape に合わせたモデル生成
    model = CNN_VAE(cnn_cfg, latent_dim, input_shape).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # CIFAR10 の画像は元々 32x32 なので、設定ファイルの input_shape に合わせてリサイズ
    resize_dims = (input_shape[1], input_shape[2])  # H x W
    # 前処理として、リサイズ後にランダム水平反転、ランダム回転などを追加
    transform = transforms.Compose([
        transforms.Resize(resize_dims),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        # 例: transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # CheckpointManager と TensorBoard の初期化（パラメータ情報を含むディレクトリ）
    checkpoint_manager = CheckpointManager(checkpoint_dir, cfg.checkpoint.topk)
    writer = SummaryWriter(log_dir=log_dir)

    def train_epoch(model, train_loader, optimizer, device, epoch):
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            loss, bce, kl = model.vae_loss(recon, data, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} Batch {batch_idx}: Loss = {loss.item():.4f}")

        avg_loss = train_loss / len(train_loader.dataset)
        print(f"====> Epoch {epoch} Average loss: {avg_loss:.4f}")
        return avg_loss

    for epoch in range(1, num_epochs + 1):
        avg_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        writer.add_scalar("Loss/train", avg_loss, epoch)
        checkpoint_manager.update(epoch, avg_loss, model)

    writer.close()

if __name__ == "__main__":
    main()
