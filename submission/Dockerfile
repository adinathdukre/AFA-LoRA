FROM --platform=linux/amd64 pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TORCH_HOME=/opt/app/resources

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user
WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install --user --no-cache-dir --no-color --requirement /opt/app/requirements.txt

COPY --chown=user:user ensemble.py inference.py /opt/app/
COPY --chown=user:user resources /opt/app/resources

ENTRYPOINT ["python", "inference.py"]
