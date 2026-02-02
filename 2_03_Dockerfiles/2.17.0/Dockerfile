FROM tensorflow/tensorflow:2.17.0-gpu-jupyter

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       graphviz \
    && rm -rf /var/lib/apt/lists/*

# Install packages in an order that ensures compatibility
RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install \
        "numpy<2.0.0" \
        "pandas==2.2.2" \
        "scipy==1.16.3" \
        "scikit-learn==1.6.1" \
        "seaborn==0.13.2" \
        "sympy==1.14.0" \
        "pydot==4.0.1" \
        "opencv-python-headless==4.12.0.88" \
    && python3 -m pip check || true

WORKDIR /tf
CMD ["/bin/bash"]