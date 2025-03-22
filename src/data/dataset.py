import os
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from PIL import Image
import torch

class CollectedImagesDataset(Dataset):
    """
    複数のエピソードディレクトリに分かれて保存された画像を読み込むための Dataset クラス。
    ディレクトリ構造の例:
        output_dir/
            ep000/
                step0000.png
                step0001.png
                ...
            ep001/
                step0000.png
                step0001.png
                ...
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): 画像が保存されているルートディレクトリ（例: config.output_dir）
            transform (callable, optional): 画像に適用する前処理
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        # os.walk を使って再帰的に PNG ファイルを探索
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(".png"):
                    self.image_paths.append(os.path.join(root, file))
        self.image_paths.sort()  # ファイルパスをソートして順序を安定させる

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

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
      - num_workers: DataLoader 用のワーカー数
      - dataset.name: "coco", "cifar10", "car_racing" のいずれか
      - dataset.root: データセットのルートディレクトリ
         coco の場合、root 内に以下の構造があることを前提とする:
           - images/train2017, images/val2017
           - annotations/instances_train2017.json, annotations/instances_val2017.json
         car_racing の場合は、収集済み画像のルートディレクトリのみ指定
      - (car_racing の場合) dataset.train_ratio: 学習／検証の分割比率（例: 0.8）
      - (car_racing の場合) dataset.seed: 分割用のシード（例: 42）
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

    dataset_name = cfg.dataset.name.lower()
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
        dataset = ImageOnlyDataset(dataset)
    elif dataset_name == "cifar10":
        train_flag = True if split == "train" else False
        dataset = datasets.CIFAR10(
            root=cfg.dataset.root,
            train=train_flag,
            download=True,
            transform=transform
        )
    elif dataset_name == "car_racing":
        # CollectedImagesDataset を使用
        dataset = CollectedImagesDataset(root_dir=cfg.dataset.root, transform=transform)
        if split in ["train", "val"]:
            train_ratio = cfg.dataset.get("train_ratio", 0.8)
            total_len = len(dataset)
            train_len = int(total_len * train_ratio)
            val_len = total_len - train_len
            # 固定シードを用いて常に同じ分割となるようにする
            g = torch.Generator()
            g.manual_seed(cfg.dataset.get("seed", 42))
            train_dataset, val_dataset = random_split(dataset, [train_len, val_len], generator=g)
            dataset = train_dataset if split == "train" else val_dataset
        # split が "test" の場合、ここでは検証と同じ扱いとするか、別途実装
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset.name}")

    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=(split == "train"), num_workers=cfg.num_workers)
    return dataloader
