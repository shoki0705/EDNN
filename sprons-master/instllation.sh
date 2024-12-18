conda create -y -n rons python==3.12
conda activate rons
conda install -y scipy
conda install -y pytorch=2.3.0 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install ipdb
pip install torchdiffeq==0.2.3
pip install torchdyn==1.0.6
pip install linear_operator==0.5.3
pip install torch-linops==0.1.3
