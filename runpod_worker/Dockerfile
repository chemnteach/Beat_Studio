FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# All HuggingFace model cache goes to the network volume
ENV HF_HOME=/workspace/hf_cache
ENV HF_HUB_CACHE=/workspace/hf_cache
ENV HF_HUB_ENABLE_HF_TRANSFER=0

COPY requirements.txt .
RUN apt-get update && apt-get install -y ffmpeg
RUN pip install --no-cache-dir -r requirements.txt

# Clone full SkyReels-V3 repo so skyreels_v3 package is importable
RUN git clone https://github.com/SkyworkAI/SkyReels-V3 /app/SkyReels-V3
# Install SkyReels-V3 deps but keep torch/torchvision from base image to avoid ABI breakage
RUN grep -vE "^(torch|torchvision|torchaudio)" /app/SkyReels-V3/requirements.txt \
    | pip install --no-cache-dir -r /dev/stdin

COPY src/ /app/src/

CMD ["python", "-u", "src/handler.py"]
