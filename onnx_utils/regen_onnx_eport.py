import argparse
import torch
from generator import define_G

# ============================================================
# Arguments
# ============================================================
parser = argparse.ArgumentParser(
    description="Export Carla2Real generator from PyTorch (.pth) to ONNX."
)

parser.add_argument(
    "--input",
    required=True,
    help="Path to input .pth checkpoint"
)

parser.add_argument(
    "--output",
    required=True,
    help="Path to output .onnx file"
)

parser.add_argument(
    "--height",
    type=int,
    required=True,
    help="Input height (must be divisible by 32)"
)

parser.add_argument(
    "--width",
    type=int,
    required=True,
    help="Input width (must be divisible by 32)"
)

args = parser.parse_args()

# ============================================================
# Config
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Build Generator
# ============================================================
def make_generator():
    return define_G(
        input_nc=3,
        output_nc=3,
        ngf=64,
        netG="global",
        norm="instance",
        n_downsample_global=4,
        n_blocks_global=9,
        n_local_enhancers=0
    ).to(device)

model = make_generator()

# ============================================================
# Load checkpoint
# ============================================================
checkpoint = torch.load(args.input, map_location=device)

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

# ============================================================
# Validate resolution
# ============================================================
if args.height % 32 != 0 or args.width % 32 != 0:
    raise ValueError(
        f"Resolution ({args.height}x{args.width}) must be divisible by 32."
    )

# ============================================================
# Dummy input
# ============================================================
dummy_input = torch.randn(
    1, 3, args.height, args.width,
    device=device
)

# ============================================================
# Export ONNX
# ============================================================
torch.onnx.export(
    model,
    dummy_input,
    args.output,
    export_params=True,
    opset_version=9,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamo=False
)

print(f"ONNX model saved to: {args.output}")
print(f"Resolution: {args.height}x{args.width}")