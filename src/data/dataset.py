import os
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

class ImageOnlyDataset(Dataset):
    """
    COCO など画像とアノテーションがセットになっているデータセットから
    画像のみを取得するためのラッパークラス。
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        return image

def get_dataloader(cfg, split="train"):
    """
    config (DictConfig) の内容に基づいて DataLoader を返す関数。
    
    必要な config のキー:
      - input_shape: [C, H, W]
      - batch_size: バッチサイズ
      - dataset.name: "coco" または "cifar10"
      - dataset.root: データセットのルートディレクトリ
         coco の場合、root 内に以下の構造があることを前提とする:
           - images/train2017, images/val2017
           - annotations/instances_train2017.json, annotations/instances_val2017.json
    """
    input_shape = tuple(cfg.input_shape)  # 例: [3, 64, 64]
    resize_dims = (input_shape[1], input_shape[2])  # H, W

    # 前処理の定義
    transform = transforms.Compose([
        transforms.Resize(resize_dims),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        # 必要に応じて Normalize など追加可能
    ])

    dataset_name = cfg.dataset.name
    if dataset_name == "coco":
        from torchvision.datasets import CocoDetection
        base_dir = cfg.dataset.root  # coco のルートディレクトリのみ指定
        if split == "train":
            img_dir = os.path.join(base_dir, "images/train2017")
            annFile = os.path.join(base_dir, "annotations/instances_train2017.json")
        elif split in ["val", "test"]:
            img_dir = os.path.join(base_dir, "images/val2017")
            annFile = os.path.join(base_dir, "annotations/instances_val2017.json")
        else:
            raise ValueError(f"Unsupported split for COCO: {split}")
        dataset = CocoDetection(root=img_dir, annFile=annFile, transform=transform)
        # 画像のみを使用するためラッパーでラベルを破棄
        dataset = ImageOnlyDataset(dataset)
    elif dataset_name == "cifar10":
        # CIFAR10 は train=True なら学習データ、False なら test データ（評価用）
        train_flag = True if split == "train" else False
        dataset = datasets.CIFAR10(
            root=cfg.dataset.root,
            train=train_flag,
            download=True,
            transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset.name}")

    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=(split == "train"))
    return dataloader
