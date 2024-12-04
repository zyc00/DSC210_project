#conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
#conda install pytorch3d -c pytorch3d -y
#pip install git+https://github.com/NVlabs/nvdiffrast.git
pip install -r requirements.txt
python setup.py build develop
#pip install -e ../detectron2
