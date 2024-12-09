# Installation
```
# require python3.10
conda create -n easyhec python=3.10
conda install cuda -c nvidia/label/cuda-12.1.1
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
pip install -r requirements.txt
pip install -e .
```
# Training and Eval
```
python tools/train_net.py -c configs/rb_solve/baxter_real/3view_pred_mask_init_prov/0and5and10.yaml model.rbsolver.use_mask anno dataset.baxter_real.load_mask_anno True
```
```
python tools/test_net.py -c configs/rb_solve/baxter_real/3view_pred_mask_init_prov/0and5and10.yaml solver.load latest
```

# Experiment Results
| Method | 2D PCK (20, 30, 40, 50) | 3D PCK (20, 30, 40, 50) |
| --- | --- | --- |
| Paper | 0.55, 0.85, 1.0, 1.0 | 0.15, 0.8, 0.9, 1.0 |
| Ours | 0.7, 0.8, 0.9, 0.9 | 0, 0.55, 0.85, 1.0 |
