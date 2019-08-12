FROM ubuntu:18.04

WORKDIR /app

RUN apt update
RUN apt install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv
RUN apt install -y git
RUN apt -y install libcairo2-dev
RUN apt install -y libsm6 libxext6 libfontconfig1 libxrender1
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
RUN git clone https://github.com/bakirillov/capsules
RUN pip install umap-learn
RUN pip install biopython
RUN apt install -y --fix-missing libpango1.0-dev
RUN pip install weasyprint
RUN pip install googledrivedownloader

RUN python3.6 download_models.py

COPY . /app

CMD ["python3.6", "cad.py"]
