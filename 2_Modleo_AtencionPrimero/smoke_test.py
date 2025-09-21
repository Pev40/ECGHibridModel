import torch
from types import SimpleNamespace

from ModeloNuevo import ECGHybridVariableBeforeBiTrans


def build_configs(num_classes=2, input_channels=12, sequence_len=512):
    cfg = SimpleNamespace()
    cfg.num_classes = num_classes
    cfg.input_channels = input_channels
    cfg.sequence_len = sequence_len
    cfg.kernel_size = 8
    cfg.stride = 1
    cfg.dropout = 0.2
    cfg.mid_channels = 32
    cfg.final_out_channels = 128
    cfg.trans_dim = 32
    cfg.num_heads = 4
    cfg.num_leads = input_channels
    return cfg


def build_hparams():
    return {"feature_dim": 128}


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = build_configs()
    hparams = build_hparams()
    print(f"Using device: {device}")
    print(f"Configs: {configs}")
    print(f"Hparams: {hparams}")
    print(f"Building model...")
    model = ECGHybridVariableBeforeBiTrans(configs, hparams).to(device)
    print(f"Model built successfully")
    print(f"Building input...")
    b = 2
    x = torch.randn(b, configs.input_channels, configs.sequence_len, device=device)
    y = model(x)
    print("Output shape:", y.shape)
    assert y.shape == (b, configs.num_classes)
    print("Smoke test OK")


if __name__ == "__main__":
    main()


