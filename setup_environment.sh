#!/bin/bash
# Use sed -i 's/\r//' setup_environment.sh to fix file endings
source ~/.bashrc

function program_is_installed {
  # set to 1 initially
  local return_=1
  # set to 0 if not found
  type $1 >/dev/null 2>&1 || { local return_=0; }
  # return value
  echo "$return_"
}

if [ $(program_is_installed conda) == 1 ]; then
  echo "Conda installed."
else
  echo "Installing Conda."
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh -q

  clear
  chmod +x miniconda.sh
  ./miniconda.sh -b -p $HOME/miniconda
  export PATH="$HOME/miniconda/bin:$PATH"
  echo "## DeepV7 ##" >> $HOME/.bashrc
  echo export PATH="$HOME/miniconda/bin:$PATH" >> $HOME/.bashrc
fi

clear

# Create an environment
conda env create -f environment.yml

clear

# Set up PYTHONPATH
echo 'export PYTHONPATH=$PWD:$PYTHONPATH' >> $HOME/.bashrc
echo "## gale ##" >> $HOME/.bashrc
echo "Setup completed!"
echo "Please run 'source ~/.bashrc' to refresh your environment"
echo "You can activate the v7labs environment with 'source activate gale'"
