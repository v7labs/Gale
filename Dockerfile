FROM nvidia/cuda:10.1-runtime
RUN groupadd -g 1001 user && \
    useradd -u 1001 -g 1001 -ms /bin/bash user && \
    mkdir /gale && \
    chown -R user:user /gale

RUN apt-get update && apt-get install -y wget

# Get miniconda and its binaries to the PATH
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    /bin/bash ./miniconda.sh -b -p /opt/conda && \
    rm ./miniconda.sh
ENV PATH /opt/conda/bin:$PATH
ENV PYTHONPATH /gale:$PYTHONPATH

#Create gale conda environment (like cd gale)
WORKDIR /gale
ADD environment.yml .
RUN conda env create -f environment.yml && conda clean -a -y

# Add the path of the python interpreter (like source activate gale)
ENV PATH /opt/conda/envs/gale/bin/:$PATH

# Copy Gale over
ADD . .
