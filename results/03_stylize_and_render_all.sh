python3 -m src.cubicstylization.main input=results/meshes/mic.stl output=results/meshes_stylized/0.33/mic.stl output_package=results/meshes_stylized/0.33/mic.pickle lambda_=0.33
python3 -m src.neus.main wandb.mode=online wandb.name=render_mic mode=render rendering.checkpoint=results/checkpoints/mic.ckpt rendering.deformation=results/meshes_stylized/0.33/mic.pickle
python3 -m src.cubicstylization.main input=results/meshes/mic.stl output=results/meshes_stylized/0.66/mic.stl output_package=results/meshes_stylized/0.66/mic.pickle lambda_=0.66
python3 -m src.neus.main wandb.mode=online wandb.name=render_mic mode=render rendering.checkpoint=results/checkpoints/mic.ckpt rendering.deformation=results/meshes_stylized/0.66/mic.pickle
python3 -m src.cubicstylization.main input=results/meshes/mic.stl output=results/meshes_stylized/1.0/mic.stl output_package=results/meshes_stylized/1.0/mic.pickle lambda_=1.0
python3 -m src.neus.main wandb.mode=online wandb.name=render_mic mode=render rendering.checkpoint=results/checkpoints/mic.ckpt rendering.deformation=results/meshes_stylized/1.0/mic.pickle
python3 -m src.cubicstylization.main input=results/meshes/mic.stl output=results/meshes_stylized/1.33/mic.stl output_package=results/meshes_stylized/1.33/mic.pickle lambda_=1.33
python3 -m src.neus.main wandb.mode=online wandb.name=render_mic mode=render rendering.checkpoint=results/checkpoints/mic.ckpt rendering.deformation=results/meshes_stylized/1.33/mic.pickle
python3 -m src.cubicstylization.main input=results/meshes/mic.stl output=results/meshes_stylized/1.67/mic.stl output_package=results/meshes_stylized/1.67/mic.pickle lambda_=1.67
python3 -m src.neus.main wandb.mode=online wandb.name=render_mic mode=render rendering.checkpoint=results/checkpoints/mic.ckpt rendering.deformation=results/meshes_stylized/1.67/mic.pickle
python3 -m src.cubicstylization.main input=results/meshes/mic.stl output=results/meshes_stylized/2.0/mic.stl output_package=results/meshes_stylized/2.0/mic.pickle lambda_=2.0
python3 -m src.neus.main wandb.mode=online wandb.name=render_mic mode=render rendering.checkpoint=results/checkpoints/mic.ckpt rendering.deformation=results/meshes_stylized/2.0/mic.pickle
python3 -m src.cubicstylization.main input=results/meshes/mic.stl output=results/meshes_stylized/2.33/mic.stl output_package=results/meshes_stylized/2.33/mic.pickle lambda_=2.33
python3 -m src.neus.main wandb.mode=online wandb.name=render_mic mode=render rendering.checkpoint=results/checkpoints/mic.ckpt rendering.deformation=results/meshes_stylized/2.33/mic.pickle
python3 -m src.cubicstylization.main input=results/meshes/mic.stl output=results/meshes_stylized/2.67/mic.stl output_package=results/meshes_stylized/2.67/mic.pickle lambda_=2.67
python3 -m src.neus.main wandb.mode=online wandb.name=render_mic mode=render rendering.checkpoint=results/checkpoints/mic.ckpt rendering.deformation=results/meshes_stylized/2.67/mic.pickle
python3 -m src.cubicstylization.main input=results/meshes/mic.stl output=results/meshes_stylized/3.0/mic.stl output_package=results/meshes_stylized/3.0/mic.pickle lambda_=3.0
python3 -m src.neus.main wandb.mode=online wandb.name=render_mic mode=render rendering.checkpoint=results/checkpoints/mic.ckpt rendering.deformation=results/meshes_stylized/3.0/mic.pickle
