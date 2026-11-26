FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    clang \
    make \
    python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN python3 -B scripts/build_and_test.py

CMD ["./build/ml-infer-cli", "benchmark", "artifacts/fraud_scorer_v1.bin", "3000"]
