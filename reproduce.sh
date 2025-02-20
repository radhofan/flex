#!/usr/bin/env bash

conda install -c conda-forge mamba -y
mamba create -n flex python=3.6.15 -y
mamba init
source ~/.bashrc
mamba activate flex
mamba install -c conda-forge r-base r-eva -y
pip install -r requirements.txt
mkdir -p flex/projects
bash flex/tool/scripts/general_setup.sh ../../projects coax-dev/coax local 37c3e667b81537768beb25bb59d0f05124624128
mamba activate coax
mamba install python=3.7.12 -y
pip install jax==0.3.25
pip install --upgrade jaxlib==0.3.22 -f https://storage.googleapis.com/jax-releases/jax_releases.html
mamba install absl-py==1.3.0 -y
pip install dm-haiku==0.0.8
pip install gym==0.25.2
pip install pandas
mamba install lz4 -y
pip install Pillow
pip install chex==0.1.5
pip install ray==2.7.2
pip install tensorboardX==2.6.2.2
pip install optax==0.1.4
mamba activate flex 
python boundschecker.py -r coax -test test_update -file flex/projects/coax/coax/experience_replay/_prioritized_test.py  -line 137 -conda coax -deps "numpy" -bc


