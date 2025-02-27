cd /workspace
git clone https://github.com/Jakur/SQAKD.git
pip install lightly
pip install tensorboard
pip install timm
pip install nvidia-dali-cuda120
pip install tensorboardX
pip install more-itertools
pip install datasets[vision]
pip install gputil
cd ./SQAKD/MQBench
python setup.py install
cd /workspace
