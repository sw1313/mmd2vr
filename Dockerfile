FROM python:3.13-slim-bookworm
ARG NUNIF_REV=92c294e6624802dd78042ebd47370a127a6d6f96
ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 GRADIO_ANALYTICS_ENABLED=False HOME=/tmp
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg ca-certificates curl libgomp1 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/nunif && curl -fsSL --retry 3 https://codeload.github.com/nagadomi/nunif/tar.gz/${NUNIF_REV} | tar -xz --strip-components=1 -C /opt/nunif
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu132 torch==2.14.1+cu132 torchvision==0.29.1+cu132 && pip install --no-cache-dir -r /opt/nunif/requirements.txt gradio==6.32.0
WORKDIR /app
ARG COMFY_REV=0072b8680dd2db0b6d68c06f718fa33bd92a728c
RUN mkdir -p /opt/ComfyUI && curl -fsSL --retry 3 https://codeload.github.com/Comfy-Org/ComfyUI/tar.gz/${COMFY_REV} | tar -xz --strip-components=1 -C /opt/ComfyUI
RUN sed -e '/^comfyui-workflow-templates/d' -e '/^comfyui-embedded-docs/d' /opt/ComfyUI/requirements.txt > /tmp/stage-requirements.txt && pip install --no-cache-dir -r /tmp/stage-requirements.txt opencv-python-headless
ENV MMD_COMFY=/opt/ComfyUI MMD_STAGE_WEIGHTS=/models/stage-weights
RUN pip install --no-cache-dir onnxruntime-gpu==1.32.0 && pip install --no-cache-dir --no-deps segment-anything==1.0 gguf==0.19.0
COPY stage_atlas.py engine.py vr180.py background_analysis.py stage_background.py stage_context.py stage_model.py stage_vr360.py vr360_*.py vr360_graph.json h3_graph.json ./
COPY comfy_nodes ./comfy_nodes
ENV MMD_INPUT=/input MMD_OUTPUT=/output MMD_STATE=/state NUNIF_HOME=/models MMD_NUNIF=/opt/nunif MMD_HOST=0.0.0.0
CMD ["python","engine.py"]
