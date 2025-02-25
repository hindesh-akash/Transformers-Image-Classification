FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

WORKDIR /app

COPY requirements.txt requirements.txt
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    pip3 install -r requirements.txt

COPY . .

CMD ["python3", "app.py"]
