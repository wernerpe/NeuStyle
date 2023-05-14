python3 -m src.cubicstylization.main input=results/meshes/teddy.stl output=results/meshes_stylized/0.95/teddy.stl output_package=results/meshes_stylized/0.95/teddy.pickle lambda_=0.95
python3 -m src.neus.main wandb.mode=online wandb.name=render_teddy mode=render rendering.checkpoint=results/checkpoints/teddy.ckpt rendering.deformation=results/meshes_stylized/0.95/teddy.pickle
python3 -m src.cubicstylization.main input=results/meshes/chair.stl output=results/meshes_stylized/0.95/chair.stl output_package=results/meshes_stylized/0.95/chair.pickle lambda_=0.95
python3 -m src.neus.main wandb.mode=online wandb.name=render_chair mode=render rendering.checkpoint=results/checkpoints/chair.ckpt rendering.deformation=results/meshes_stylized/0.95/chair.pickle
python3 -m src.cubicstylization.main input=results/meshes/drums.stl output=results/meshes_stylized/0.95/drums.stl output_package=results/meshes_stylized/0.95/drums.pickle lambda_=0.95
python3 -m src.neus.main wandb.mode=online wandb.name=render_drums mode=render rendering.checkpoint=results/checkpoints/drums.ckpt rendering.deformation=results/meshes_stylized/0.95/drums.pickle
python3 -m src.cubicstylization.main input=results/meshes/ficus.stl output=results/meshes_stylized/0.95/ficus.stl output_package=results/meshes_stylized/0.95/ficus.pickle lambda_=0.95
python3 -m src.neus.main wandb.mode=online wandb.name=render_ficus mode=render rendering.checkpoint=results/checkpoints/ficus.ckpt rendering.deformation=results/meshes_stylized/0.95/ficus.pickle
python3 -m src.cubicstylization.main input=results/meshes/hotdog.stl output=results/meshes_stylized/0.95/hotdog.stl output_package=results/meshes_stylized/0.95/hotdog.pickle lambda_=0.95
python3 -m src.neus.main wandb.mode=online wandb.name=render_hotdog mode=render rendering.checkpoint=results/checkpoints/hotdog.ckpt rendering.deformation=results/meshes_stylized/0.95/hotdog.pickle
python3 -m src.cubicstylization.main input=results/meshes/lego.stl output=results/meshes_stylized/0.95/lego.stl output_package=results/meshes_stylized/0.95/lego.pickle lambda_=0.95
python3 -m src.neus.main wandb.mode=online wandb.name=render_lego mode=render rendering.checkpoint=results/checkpoints/lego.ckpt rendering.deformation=results/meshes_stylized/0.95/lego.pickle
python3 -m src.cubicstylization.main input=results/meshes/materials.stl output=results/meshes_stylized/0.95/materials.stl output_package=results/meshes_stylized/0.95/materials.pickle lambda_=0.95
python3 -m src.neus.main wandb.mode=online wandb.name=render_materials mode=render rendering.checkpoint=results/checkpoints/materials.ckpt rendering.deformation=results/meshes_stylized/0.95/materials.pickle
python3 -m src.cubicstylization.main input=results/meshes/mic.stl output=results/meshes_stylized/0.95/mic.stl output_package=results/meshes_stylized/0.95/mic.pickle lambda_=0.95
python3 -m src.neus.main wandb.mode=online wandb.name=render_mic mode=render rendering.checkpoint=results/checkpoints/mic.ckpt rendering.deformation=results/meshes_stylized/0.95/mic.pickle
python3 -m src.cubicstylization.main input=results/meshes/ship.stl output=results/meshes_stylized/0.95/ship.stl output_package=results/meshes_stylized/0.95/ship.pickle lambda_=0.95
python3 -m src.neus.main wandb.mode=online wandb.name=render_ship mode=render rendering.checkpoint=results/checkpoints/ship.ckpt rendering.deformation=results/meshes_stylized/0.95/ship.pickle
