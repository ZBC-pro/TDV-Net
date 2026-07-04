import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

try:
    import open_clip
    REMOTECLIP_AVAILABLE = True
    print(">> open_clip library detected. TGSI module enabled.")
except ImportError:
    REMOTECLIP_AVAILABLE = False
    print(">> WARNING: open_clip not installed. TGSI will use random embeddings.")


class TGSI(nn.Module):
    def __init__(
        self,
        vis_channels: int,
        embed_dim: int = 512,
        sod_classes: Optional[List[str]] = None,
        floe_classes: Optional[List[str]] = None,
        prompt_template: str = "A SAR image of {}",
        remoteclip_model: str = "ViT-L-14",
        remoteclip_pretrained: Optional[str] = None,
        freeze_text_encoder: bool = True,
        tau_init: float = 0.07
    ):
        super().__init__()

        if sod_classes is None or len(sod_classes) == 0:
            raise ValueError("TGSI requires 'tgsi_sod_classes' in config.")
        if floe_classes is None or len(floe_classes) == 0:
            raise ValueError("TGSI requires 'tgsi_floe_classes' in config.")

        self.embed_dim = embed_dim
        self.sod_classes = sod_classes
        self.floe_classes = floe_classes
        self.k_sod = len(self.sod_classes)
        self.k_floe = len(self.floe_classes)
        self.k_total = self.k_sod + self.k_floe
        self.prompt_template = prompt_template

        print(f">> TGSI initialized with {self.k_sod} SOD classes and {self.k_floe} FLOE classes")

        self.vis_proj = nn.Sequential(
            nn.Conv2d(vis_channels, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False)
        )

        self.tau = nn.Parameter(torch.tensor(tau_init, dtype=torch.float32))
        self.text_embedding = nn.Embedding(self.k_total, embed_dim)
        self.text_embedding.weight.requires_grad = not freeze_text_encoder

        self._init_text_embeddings(remoteclip_model, remoteclip_pretrained, embed_dim)
        self.out_channels = vis_channels + self.k_total

    def _init_text_embeddings(self, model_name: str, pretrained_path: Optional[str], embed_dim: int):
        text_features = None

        if REMOTECLIP_AVAILABLE and pretrained_path is not None:
            import os
            if os.path.exists(pretrained_path):
                print(f">> Loading RemoteCLIP: {model_name} from {pretrained_path}")
                try:
                    clip_model, _, _ = open_clip.create_model_and_transforms(
                        model_name, pretrained=pretrained_path
                    )
                    tokenizer = open_clip.get_tokenizer(model_name)

                    all_prompts = []
                    for cls_name in self.sod_classes:
                        all_prompts.append(self.prompt_template.format(cls_name))
                    for cls_name in self.floe_classes:
                        all_prompts.append(self.prompt_template.format(cls_name))

                    clip_model = clip_model.cpu().float().eval()
                    with torch.no_grad():
                        tokens = tokenizer(all_prompts)
                        text_features = clip_model.encode_text(tokens)
                        text_features = F.normalize(text_features, p=2, dim=-1)
                        text_features = text_features.float()

                    del clip_model, tokenizer
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f">> ERROR loading RemoteCLIP: {e}")
                    text_features = None

        if text_features is not None:
            if text_features.shape[-1] != embed_dim:
                raise ValueError(
                    f"RemoteCLIP text feature dimension ({text_features.shape[-1]}) does not match embed_dim ({embed_dim})."
                )
            with torch.no_grad():
                self.text_embedding.weight.copy_(text_features)
            print(">> Text embeddings loaded from RemoteCLIP")
        else:
            print(">> WARNING: Using random text embeddings")
            with torch.no_grad():
                nn.init.normal_(self.text_embedding.weight, mean=0, std=0.02)

    def forward(self, f_agg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = f_agg.shape
        device = f_agg.device
        dtype = f_agg.dtype

        f_vis = self.vis_proj(f_agg)
        f_vis_norm = F.normalize(f_vis, p=2, dim=1)
        f_vis_flat = f_vis_norm.permute(0, 2, 3, 1).reshape(B, H * W, -1)

        indices = torch.arange(self.k_total, device=device)
        text_anchors = self.text_embedding(indices).to(dtype=dtype)
        text_anchors_norm = F.normalize(text_anchors, p=2, dim=-1)

        similarity = torch.matmul(f_vis_flat, text_anchors_norm.t())
        tau_clamped = self.tau.clamp(min=0.01)
        s_score = F.softmax(similarity / tau_clamped, dim=-1)
        s_score = s_score.view(B, H, W, self.k_total).permute(0, 3, 1, 2)

        f_injected = torch.cat([f_agg, s_score], dim=1)
        return f_injected, s_score


if __name__ == "__main__":
    pass
