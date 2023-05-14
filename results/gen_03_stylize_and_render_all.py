from pathlib import Path

LAMBDA = 0.95
SCENES = [
    "teddy",
    "chair",
    "drums",
    "ficus",
    "hotdog",
    "lego",
    "materials",
    "mic",
    "ship",
]

with Path("results/03_stylize_and_render_all.sh").open("w") as f:
    for scene in SCENES:
        f.write(
            f"""python3 -m src.cubicstylization.main input=results/meshes/{scene}.stl output=results/meshes_stylized/{LAMBDA}/{scene}.stl output_package=results/meshes_stylized/{LAMBDA}/{scene}.pickle lambda_={LAMBDA}
python3 -m src.neus.main wandb.mode=online wandb.name=render_{scene} mode=render rendering.checkpoint=results/checkpoints/{scene}.ckpt rendering.deformation=results/meshes_stylized/{LAMBDA}/{scene}.pickle
"""
        )
