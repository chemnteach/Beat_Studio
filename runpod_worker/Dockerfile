FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# All HuggingFace model cache goes to the network volume
ENV HF_HOME=/runpod-volume/hf_cache
ENV HF_HUB_CACHE=/runpod-volume/hf_cache

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/

CMD ["python", "-u", "src/handler.py"]
