from pathlib import Path

LAMBDAS = [0.33, 0.66, 1.0, 1.33, 1.67, 2.0, 2.33, 2.67, 3.0]
SCENES = [
    "mic",
]
scene = "mic"

with Path("results/03_stylize_and_render_all.sh").open("w") as f:
    for lval in LAMBDAS:
        f.write(
            f"""python3 -m src.cubicstylization.main input=results/meshes/{scene}.stl output=results/meshes_stylized/{lval}/{scene}.stl output_package=results/meshes_stylized/{lval}/{scene}.pickle lambda_={lval}
python3 -m src.neus.main wandb.mode=online wandb.name=render_{scene}_{lval} mode=render rendering.checkpoint=results/checkpoints/{scene}.ckpt rendering.deformation=results/meshes_stylized/{lval}/{scene}.pickle
"""
        )
