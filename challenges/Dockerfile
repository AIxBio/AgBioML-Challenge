FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    wget

# Get conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda
# Remove the downloaded installer
RUN rm Miniconda3-latest-Linux-x86_64.sh

# Add conda to the PATH environment variable for all subsequent RUN instructions
ENV PATH=/opt/conda/bin:$PATH

# Now conda commands should work directly
# First, define an environment with scikit-learn
RUN conda install -c bioconda -c conda-forge -c pytorch python=3.12 \
 scikit-learn pandas seaborn pandas \
 pytorch::pytorch pytorch::torchvision \
 scanpy rpy2 r-base bioconda::bioconductor-minfi

RUN conda install -c bioconda -c conda-forge bioconda::bioconductor-sesame pyarrow

