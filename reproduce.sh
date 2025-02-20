#!/usr/bin/env bash

# Install Miniconda
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda

# Set environment variables
export PATH="$HOME/miniconda/bin:$PATH"
export MAMBA_ROOT_PREFIX="$HOME/miniconda"

# Install Mamba
conda install -c conda-forge mamba -y

# Initialize Mamba shell
mamba shell init --shell=bash
source ~/.bashrc  # Reload shell config
eval "$(mamba shell hook --shell=bash)"

# Create and activate 'flex' environment
mamba create -n flex python=3.6.15 -c conda-forge -y
mamba activate flex

# Install dependencies
mamba install -c conda-forge r-base r-eva -y
mamba install hdf5 -y
mamba install libnetcdf -y
pip install -r flex/requirements.txt

# Ensure the projects directory exists
mkdir -p flex/projects

# Run setup script
bash flex/tool/scripts/general_setup.sh flex/projects coax-dev/coax local 37c3e667b81537768beb25bb59d0f05124624128

# Activate 'coax' environment and install dependencies
mamba create -n coax python=3.7.12 -c conda-forge -y
mamba activate coax
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

# Activate 'flex' and run the boundschecker
mamba activate flex
python flex/tool/boundschecker.py -r coax -test test_update -file flex/projects/coax/coax/experience_replay/_prioritized_test.py -line 137 -conda coax -deps "numpy" -bc
