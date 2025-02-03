# neurotools
Tools useful for analysis of all different sorts of data, along with applications

## Getting started
- clone the repository to your machine
  - make sure git is installed on your computer
  - copy the link in the code dropdown menu
  - change your working directory to where you want to put the project
  - in a terminal / power shell / command prompt, run `git clone LINK_YOU_COPIED`
  - This will create a directory called `neurotools` that contains all of the project files, which we will call `REPO_ROOT`
  - Change working directory to `REPO_ROOT`
  - _These steps are also doable via the graphical GitHub interface on windows_
- Environment / dependencies setup is easiest with anaconda package manager. Make sure you have anaconda or miniconda installed and added to system path.
  - _On windows installing anaconda also install a conda shell, which can run unix like conda commands. On a unix-like system, conda will be availble in your terminal_
- Create a new environment with `conda create -n ENV_NAME`
- Activate the environment with `conda activate ENV_NAME`
- To install all required packages in your environment, run `conda install --file REPO_ROOT/requirements.txt`
  - where REPO_ROOT is the path to the root of the neurotools repository.
  - note that this will install the CPU only verison of pytorch. See pytorch's "get started locally" page for instructions on how to set up CUDA and install the GPU version. Some algorithms will run very slowly on CPU only.
- You can install neurotools itself as a package in this conda environment, so that it is accessable in other projects, via `pip install -e REPO_ROOT`
  - note this isn't necessary for running self-contained code / examples.

## Project Structure
All core code is contained in the neurotools subdirectory. There are 7 main modules.
- `utils` contains utility code for functions like indexing arrays, sampling data, and other important things that are used by multiple functions.
- `stats` contains standalone functions for estimating statistics on data.
- `modules` contains pytorch style modules (inherits from `torch.nn.Modules`) that can be used as components of machine learning architectures.
  - This includes the `VarConvND` layer that is critical for the function of the layered searchlight.
- `decoding` includes (vaguely) sklearn style full decoding models that can take dataloaders as input and produce predictions / other fits to data.
  - So far contains only the ROISearhlightDecoder (i.e. layered searchlight with ROI based niceties.)
- `geometry` contains tools for estimating the representational structure of stuff in data.
- `embed` contains (vaguely) sklearn style classes for dimensionality reduction / visualization
  - Includes a Multidimmensional Scaling Tool and a full implimentation of Sparse Supervised Embedding.

## Running Searchlight Simulator example notebook
- In the REPO_ROOT directory, with your conda environment ENV_NAME activated, run `jupyter notebook`. This should open and interactive session in your defualt web browser.
- Navigate to `examples` and open the `searchlight_simulation.ipynb` file. 
- You can run through the cells for a quick tutorial on using the layered searchlight model and the MDS tool.
