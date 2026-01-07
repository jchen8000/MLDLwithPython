FROM tensorflow/tensorflow:2.19.0-gpu-jupyter

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get -y upgrade \
    && apt-get install -y --no-install-recommends \
       graphviz \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install -r /tmp/requirements.txt \
    && python3 -m pip check || true

WORKDIR /tf
CMD ["/bin/bash"]
