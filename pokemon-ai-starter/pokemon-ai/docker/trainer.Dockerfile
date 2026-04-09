FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends     python3 python3-pip git curl tini && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["python3","src/train.py"]
