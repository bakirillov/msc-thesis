FROM ubuntu:18.04

WORKDIR /app

RUN apt update
RUN apt install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv
RUN apt install -y git
RUN apt -y install libcairo2-dev
RUN apt-get install -y wget bzip2
RUN apt-get -y install sudo
RUN apt install -y libsm6 libxext6 libfontconfig1 libxrender1 wget
RUN python3.6 -m pip install pip --upgrade
RUN python3.6 -m pip install wheel
RUN pip install --upgrade setuptools
RUN pip install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
RUN pip install torchvision
RUN pip install Flask
RUN pip install dash==0.39
RUN pip install vedis
RUN pip install scikit-image
RUN pip install -U scikit-learn
RUN pip install tpot
RUN pip install catboost
RUN git clone https://github.com/bakirillov/capsules
RUN pip install umap-learn
RUN pip install biopython
RUN apt install -y --fix-missing libpango1.0-dev
RUN pip install weasyprint
RUN pip install googledrivedownloader
RUN adduser --disabled-password --gecos '' ubuntu
RUN adduser ubuntu sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER ubuntu
WORKDIR /home/ubuntu/
RUN chmod a+rwx /home/ubuntu/
RUN wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
RUN bash Anaconda3-5.0.1-Linux-x86_64.sh -b
RUN rm Anaconda3-5.0.1-Linux-x86_64.sh
ENV PATH /home/ubuntu/anaconda3/bin:$PATH
RUN conda install -c bioconda weblogo
WORKDIR /app

COPY . /app
RUN python3.6 ./download_models.py

CMD ["python3.6", "cad.py"]
