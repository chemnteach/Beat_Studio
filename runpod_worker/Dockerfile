FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# All HuggingFace model cache goes to the network volume (mounted at /runpod-volume)
ENV HF_HOME=/runpod-volume/hf_cache
ENV HF_HUB_CACHE=/runpod-volume/hf_cache
ENV HF_HUB_ENABLE_HF_TRANSFER=0

# Execution timeout hint — ALSO set "Max Execution Time" to 3600 in the
# RunPod endpoint dashboard (Serverless → your endpoint → Edit → Advanced)
ENV RUNPOD_MAX_JOB_TIMEOUT=3600

COPY requirements.txt .
RUN apt-get update && apt-get install -y ffmpeg
# Install PyTorch 2.6 first for Blackwell (sm_120) GPU support
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
# Install remaining deps, excluding torch/torchvision/torchaudio (exact match, not torchao)
RUN grep -vE "^(torch|torchvision|torchaudio)$" requirements.txt \
    | pip install --no-cache-dir -r /dev/stdin
RUN pip install --no-cache-dir boto3

# Clone full SkyReels-V3 repo so skyreels_v3 package is importable
RUN git clone https://github.com/SkyworkAI/SkyReels-V3 /app/SkyReels-V3
# Install SkyReels-V3 deps but keep torch/torchvision from base image to avoid ABI breakage
RUN grep -vE "^(torch|torchvision|torchaudio)" /app/SkyReels-V3/requirements.txt \
    | pip install --no-cache-dir -r /dev/stdin

COPY src/ /app/src/

CMD ["python", "-u", "src/handler.py"]
