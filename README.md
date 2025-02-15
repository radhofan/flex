### NOTE: THIS REPO IS INTENDED AS A REPRODUCIBLE PAPER EXPERIMENT AND IS INTENDED TO BE RAN INSIDE A TERMINAL IN EITHER TROVI/BINDER
### THIS README.MD MAY HAVE MORE INSTRUCTIONS THAN THE ORIGINAL REPO TO ENSURE PERFECT REPLICATION
### REPRODUCE THIS REPO IN TROVI VIA THIS LINK: https://chameleoncloud.org/experiment/share/6af4bcce-5c57-4051-83e3-122647110c9a
### REPRODUCE THIS REPO IN BINDER VIA THIS LINK: https://mybinder.org/ 
    - GitHub Repository: `[Insert GitHub Repo Link Here]`  
    - Notebook File (`.ipynb`):** `[Insert Notebook File Name Here]`  
### ORIGINAL REPO LINK: https://github.com/uiuc-arc/flex

# FLEX: Fixing Flaky tests in Machine Learning Projects 

This repository provides an implementation of our paper: [FLEX: Fixing Flaky tests in Machine Learning Projects by Updating Assertion Bounds](http://misailo.web.engr.illinois.edu/papers/flex-fse21.pdf). 

FLEX uses Extreme Value Theory to determine appropriate bounds for assertions in tests for stochastic Machine Learning algorithms.

## Set up Trovi Project (if using trovi) :

cd into correct working directory

Deactivate default conda trovi environment
```bash
conda deactivate
```

### Conda environment
Make a new conda mamba environment with python 3.6.15
```bash
mamba create -n flex python=3.6.15
mamba init
source ~/.bashrc
mamba activate flex
```

### Conda setup
We recommend using a conda environment to install and use flex.
Run `mamba install -c conda-forge r-base r-eva` to installed the required R packages

### Python Setup
Use Python (3.6-3.8) version due to issues in [astunparse library](https://github.com/simonpercivall/astunparse/issues/62)  
To install the requirements, do `pip install -r requirements.txt`.

To install individual projects which contain flaky tests, use the `scripts/general_setup.sh` script. See the details in next section.

## Running Flex

Step 1: Create `projects` directory in root.

Step 2; Replace the source in all .sh files with:
```bash
source /opt/conda/etc/profile.d/conda.sh
```

Step 3: Go to `tool/scripts` and run `bash general_setup.sh ../../projects [github-slug] [local/global] [commit]` to set up the project.

E.g, for coax:
`bash general_setup.sh ../../projects coax-dev/coax local 37c3e667b81537768beb25bb59d0f05124624128`

For coax: We need to install some dependencies manually since there are some errors in the original commit provided
```bash
mamba activate coax
mamba install python=3.7.12
pip install jax==0.3.25
pip install --upgrade jaxlib==0.3.22 -f https://storage.googleapis.com/jax-releases/jax_releases.html
mamba install absl-py==1.3.0
pip install dm-haiku==0.0.8
pip install gym==0.25.2
pip install pandas
mamba install lz4
pip install Pillow
pip install chex==0.1.5
pip install ray==2.7.2
pip install tensorboardX==2.6.2.2
pip install optax==0.1.4
mamba activate flex
```

For coax: there are some error regarding jax not having the `jax.tree_multimap` module, it should be replace by the `jax.tree` module instead, the files are in coax/utils/_array.py and coax/experience_replay/_prioritized.py.

All slugs and project commits used in the paper can be found in `newbugs.csv`. `global` mode will install some system level dependencies required for some projects, may need sudo access. Use `local` to avoid installing them.

Step 4: Run `python boundschecker.py -r [repo_name] -test [test_name] -file [filename]  -line [line number] -conda [conda env name] -bc (enables boxcox transformation)` in the `tool/` directory to run FLEX for the project.

E.g., for coax:
`python boundschecker.py -r coax -test test_update -file coax/coax/experience_replay/_prioritized_test.py  -line 137 -conda coax -deps "numpy" -bc`
This will produce output like...
```
Bound: 0.000794638226562553
Expected: 0.001
Expected is greater than lower bound: 0.001 >= 0.000794638226562553
Patch: Generating looser patch
<location of patch>
Diff generated:
<location of diff>
```

## Explanation of flags

- Repo Name: -r
- Test Name: -test
- File name: -file
- Class name: -cl
- Line number of assertion: -line
- Conda env name: -conda
- Enable Box-Cox optimization: -bc
- Number of threads: -t (default 1)

## Directory Structure

The source code for the project is mainly contained in the `tool/` directory. The `tool/` directory is further split into sub-directories like `src` which contains implementation files, folders with setup scripts (`scripts`), logs folders and other implementation files. The root directory further contains some top level files like `requirements.txt`.

## FLEX Configuration

The file `src/Config.py` contains all the configurations for the tool

- DEFAULT_ITERATIONS: Number of samples to collect in first round (50)
- SUBSEQUENT_ITERATIONS: Number of samples to collect in subsequent rounds (50)
- MAX_ITERATIONS: Max samples (1000)
- THREAD_COUNT: Number of threads (1)
- USE_BOXCOX: Apply boxcox transformation (False)
- BOUNDS_MAX_PERCENTILE: Max percentile to check (0.9999)
- MIN_TAIL_VALUES: Minimum tail values (50)

## Flaky Tests used in the paper

The file `newbugs.csv` contains the list of flaky test we used in our paper. Each row presents the details: filename, classname, filename, and assert line. It also contains the commit id of the repository that we used.


## Citation

If you use our tool, please cite us using:
```
@inproceedings{dutta2021flex,
  title={FLEX: Fixing Flaky Tests in Machine Learning Projects by Updating Assertion Bounds},
  author={Dutta, Saikat and Shi, August and Misailovic, Sasa},
  year={2021},
  organization={FSE}
}
```

