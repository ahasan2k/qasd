FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml README.md ./
COPY qasd ./qasd
RUN pip install --no-cache-dir .
COPY config ./config
RUN useradd --create-home qasd && mkdir -p /data && chown qasd /data
USER qasd
EXPOSE 8787
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8787/healthz')"
CMD ["python", "-m", "qasd"]
