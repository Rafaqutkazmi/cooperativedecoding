FROM python:3.11-slim
WORKDIR /workspace/tomm-cooperative-decoder-public
COPY environment/requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock
COPY . /workspace/tomm-cooperative-decoder-public
CMD ["python3", "-m", "unittest", "discover", "-s", "tests", "-v"]
