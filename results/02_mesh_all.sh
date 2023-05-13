# Before running this script, ensure that checkpoints have been downloaded from wandb and put in 
python3 -m src.neus.main mode=mesh meshing.checkpoint=results/checkpoints/teddy.ckpt meshing.path=results/meshes/teddy.stl
python3 -m src.neus.main mode=mesh meshing.checkpoint=results/checkpoints/chair.ckpt meshing.path=results/meshes/chair.stl
python3 -m src.neus.main mode=mesh meshing.checkpoint=results/checkpoints/drums.ckpt meshing.path=results/meshes/drums.stl
python3 -m src.neus.main mode=mesh meshing.checkpoint=results/checkpoints/ficus.ckpt meshing.path=results/meshes/ficus.stl
python3 -m src.neus.main mode=mesh meshing.checkpoint=results/checkpoints/hotdog.ckpt meshing.path=results/meshes/hotdog.stl
python3 -m src.neus.main mode=mesh meshing.checkpoint=results/checkpoints/lego.ckpt meshing.path=results/meshes/lego.stl
python3 -m src.neus.main mode=mesh meshing.checkpoint=results/checkpoints/materials.ckpt meshing.path=results/meshes/materials.stl
python3 -m src.neus.main mode=mesh meshing.checkpoint=results/checkpoints/mic.ckpt meshing.path=results/meshes/mic.stl
python3 -m src.neus.main mode=mesh meshing.checkpoint=results/checkpoints/ship.ckpt meshing.path=results/meshes/ship.stl
