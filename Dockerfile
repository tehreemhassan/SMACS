FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY src /app/src
COPY run_scenarios.py /app/
RUN mkdir outputs
CMD ["python", "run_scenarios.py"]
