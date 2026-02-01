"""OSNet-based ReID extractor for player embeddings."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.vision.reid.base import ReIDExtractor


class OSNetExtractor(ReIDExtractor):
    """
    OSNet-based person re-identification embedding extractor.

    Uses OSNet-x0.25 (~2MB), a lightweight but effective ReID model.
    Supports MPS (Apple Silicon), CUDA, and CPU backends.
    """

    HUGGINGFACE_REPO = "mikel-brostrom/osnet_x0_25_msmt17_256x128_amsgrad"
    WEIGHTS_FILENAME = "osnet_x0_25_msmt17_256x128_amsgrad_ep180_stp80_acc83.92_ema.pth.tar-90"

    def __init__(
        self,
        model_name: str = "osnet_x0_25",
        device: str = "mps",
        crop_size: tuple[int, int] = (256, 128),  # height x width
        batch_size: int = 32,
        cache_dir: str = "models",
    ):
        """
        Initialize OSNet extractor.

        Args:
            model_name: OSNet variant name (currently only 'osnet_x0_25' supported).
            device: Device to run on ('mps', 'cuda', 'cpu').
            crop_size: Input crop size (height, width).
            batch_size: Batch size for inference.
            cache_dir: Directory to cache downloaded weights.
        """
        self.model_name = model_name
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Select device
        if device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Load model
        self.model = self._load_model()
        self.model.eval()

        # Define preprocessing transform (ImageNet normalization)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(crop_size),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimensionality (512 for OSNet)."""
        return 512

    def _load_model(self) -> torch.nn.Module:
        """Load OSNet model with pretrained weights."""
        # Build OSNet architecture
        model = self._build_osnet_x0_25()

        # Download and load weights
        weights_path = self._download_weights()
        if weights_path.exists():
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            # Remove 'module.' prefix if present (from DataParallel training)
            state_dict = {
                k.replace("module.", ""): v
                for k, v in state_dict.items()
            }

            model.load_state_dict(state_dict, strict=False)

        return model.to(self.device)

    def _download_weights(self) -> Path:
        """Download OSNet weights from HuggingFace if not cached."""
        weights_path = self.cache_dir / f"{self.model_name}_msmt17.pth"

        if weights_path.exists():
            return weights_path

        try:
            from huggingface_hub import hf_hub_download

            downloaded_path = hf_hub_download(
                repo_id=self.HUGGINGFACE_REPO,
                filename=self.WEIGHTS_FILENAME,
                cache_dir=str(self.cache_dir),
            )
            # Copy to our standard path
            import shutil
            shutil.copy(downloaded_path, weights_path)
            return weights_path

        except ImportError:
            print("Warning: huggingface_hub not installed. Using random weights.")
            return weights_path
        except Exception as e:
            print(f"Warning: Failed to download weights: {e}. Using random weights.")
            return weights_path

    def _build_osnet_x0_25(self) -> torch.nn.Module:
        """Build OSNet-x0.25 architecture."""
        # OSNet-x0.25 has width multiplier of 0.25
        return OSNet(
            num_classes=1,  # We only need features, not classification
            feature_dim=512,
            width_mult=0.25,
        )

    def extract(self, crops: list[np.ndarray]) -> np.ndarray:
        """
        Extract embeddings from a batch of person crops.

        Args:
            crops: List of RGB images (H, W, 3) as numpy arrays.

        Returns:
            Embeddings array of shape (N, 512).
        """
        if len(crops) == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        # Preprocess crops
        tensors = []
        for crop in crops:
            # Ensure RGB format (H, W, 3)
            if crop.ndim == 2:
                crop = np.stack([crop] * 3, axis=-1)

            # Apply transform
            tensor = self.transform(crop)
            tensors.append(tensor)

        # Stack into batch
        batch = torch.stack(tensors).to(self.device)

        # Extract features in batches
        all_features = []

        with torch.no_grad():
            for i in range(0, len(batch), self.batch_size):
                batch_slice = batch[i : i + self.batch_size]
                features = self.model(batch_slice)
                # L2 normalize embeddings
                features = F.normalize(features, p=2, dim=1)
                all_features.append(features.cpu().numpy())

        return np.concatenate(all_features, axis=0)


# OSNet architecture implementation (simplified)
class ConvBlock(torch.nn.Module):
    """Basic convolution block with batch norm and ReLU."""

    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_c)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class LiteConv(torch.nn.Module):
    """Lightweight 1x1 convolution."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class ChannelGate(torch.nn.Module):
    """Channel attention gate."""

    def __init__(self, num_channels, reduction=16):
        super().__init__()
        mid_c = num_channels // reduction
        self.fc1 = torch.nn.Conv2d(num_channels, mid_c, 1)
        self.fc2 = torch.nn.Conv2d(mid_c, num_channels, 1)

    def forward(self, x):
        gap = F.adaptive_avg_pool2d(x, 1)
        gate = torch.sigmoid(self.fc2(F.relu(self.fc1(gap))))
        return x * gate


class OSBlock(torch.nn.Module):
    """Omni-scale building block."""

    def __init__(self, in_c, out_c, reduction=4):
        super().__init__()
        mid_c = out_c // reduction

        self.conv1 = LiteConv(in_c, mid_c)

        # Multi-scale streams
        self.conv2a = LiteConv(mid_c, mid_c)
        self.conv2b = torch.nn.Sequential(
            LiteConv(mid_c, mid_c),
            LiteConv(mid_c, mid_c),
        )
        self.conv2c = torch.nn.Sequential(
            LiteConv(mid_c, mid_c),
            LiteConv(mid_c, mid_c),
            LiteConv(mid_c, mid_c),
        )
        self.conv2d = torch.nn.Sequential(
            LiteConv(mid_c, mid_c),
            LiteConv(mid_c, mid_c),
            LiteConv(mid_c, mid_c),
            LiteConv(mid_c, mid_c),
        )

        self.gate = ChannelGate(mid_c)
        self.conv3 = LiteConv(mid_c, out_c)

        # Residual
        self.downsample = None
        if in_c != out_c:
            self.downsample = LiteConv(in_c, out_c)

    def forward(self, x):
        identity = x

        x1 = self.conv1(x)

        # Multi-scale aggregation
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)

        x2 = self.gate(x2a + x2b + x2c + x2d)
        x3 = self.conv3(x2)

        if self.downsample is not None:
            identity = self.downsample(identity)

        return F.relu(x3 + identity)


class OSNet(torch.nn.Module):
    """
    Omni-Scale Network for person re-identification.

    Reference: Zhou et al. "Omni-Scale Feature Learning for Person Re-Identification"
    """

    def __init__(
        self,
        num_classes: int = 1,
        feature_dim: int = 512,
        width_mult: float = 1.0,
    ):
        super().__init__()

        # Channel configuration (scaled by width_mult)
        channels = [64, 256, 384, 512]
        channels = [int(c * width_mult) for c in channels]

        # Stem
        self.conv1 = ConvBlock(3, channels[0], 7, 2, 3)
        self.maxpool = torch.nn.MaxPool2d(3, 2, 1)

        # OSNet blocks
        self.conv2 = torch.nn.Sequential(
            OSBlock(channels[0], channels[1]),
            OSBlock(channels[1], channels[1]),
        )
        self.pool2 = torch.nn.Sequential(
            ConvBlock(channels[1], channels[1], 1, 1, 0),
            torch.nn.AvgPool2d(2, 2),
        )

        self.conv3 = torch.nn.Sequential(
            OSBlock(channels[1], channels[2]),
            OSBlock(channels[2], channels[2]),
        )
        self.pool3 = torch.nn.Sequential(
            ConvBlock(channels[2], channels[2], 1, 1, 0),
            torch.nn.AvgPool2d(2, 2),
        )

        self.conv4 = torch.nn.Sequential(
            OSBlock(channels[2], channels[3]),
            OSBlock(channels[3], channels[3]),
        )

        # Final 1x1 conv to get feature_dim
        self.conv5 = torch.nn.Conv2d(channels[3], feature_dim, 1)
        self.gap = torch.nn.AdaptiveAvgPool2d(1)

        self.fc = torch.nn.Linear(feature_dim, num_classes) if num_classes > 0 else None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.conv5(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        return x
