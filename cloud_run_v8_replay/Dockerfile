FROM --platform=linux/amd64 python:3.12.13-slim-bookworm@sha256:6e13e65c55e33adf203d77ee371cf8bf5d81bd4902ef07565721f46bf44917af

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/theseus
COPY . /opt/theseus

RUN python -m pip install --no-cache-dir --disable-pip-version-check \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple \
      -r requirements.txt \
    && python -m pip check \
    && groupadd --gid 10001 theseus \
    && useradd --uid 10001 --gid 10001 --home-dir /home/theseus --create-home --shell /usr/sbin/nologin theseus \
    && mkdir -p /Users/shaanp/.codex/custody/theseus-lab-sandbox-v1-replacement-loss-v6 \
    && chown -R 10001:10001 /Users/shaanp/.codex /home/theseus

USER 10001:10001
ENTRYPOINT ["python", "/opt/theseus/cloud_run_entrypoint.py"]
