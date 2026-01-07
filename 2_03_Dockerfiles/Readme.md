
# 🚀 Build and Publish the Docker Image

This guide explains how to build a Docker image and push it to Docker Hub.  


---

## ✅ 1. Files in This Folder
- **Dockerfile** – Defines the image build steps.
- **requirements.txt** – Lists pinned Python packages for reproducibility.

## ✅ 2. Prerequisites
- Docker installed on your machine.
- (optionsl) Docker Hub account and credentials.
- (optional) NVIDIA drivers and Docker GPU runtime for GPU support.


## ✅ 3. Log in to Docker Hub (Optional)
```
docker login
```
Enter your Docker Hub username and password.
Verify login:
```
docker info | grep Username
```

## ✅ 4. Build the Docker Image
Navigate to the folder containing your Dockerfile and requirements.txt.

Build the image
```
docker build -t dockerhub-username/tf_gpu_jupyterlab:2.19.0 .
```

## ✅ 5. Test the image locally
```
docker run --gpus all -p 8888:8888 \
  -v /path/to/your/code:/tf/workspace \
  -it --rm dockerhub-username/tf_gpu_jupyterlab:2.19.0 bash
```

Verify the version of the packages
```
pip list
```

Start JupyterLab:
```
jupyter-lab --notebook-dir=/tf/workspace \
  --ip 0.0.0.0 --no-browser --allow-root \
  --port=8888 --NotebookApp.port_retries=0
```


## ✅ 6. Tag as latest (Optional)
```
docker tag dockerhub-username/tf_gpu_jupyterlab:2.19.0 \
           dockerhub-username/tf_gpu_jupyterlab:latest
```

## ✅ 7. Push to Docker Hub (Optional)
```
docker push dockerhub-username/tf_gpu_jupyterlab:2.19.0
docker push dockerhub-username/tf_gpu_jupyterlab:latest
```

## ✅ 8. Logout (Optional)
```
docker logout
```