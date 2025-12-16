
FROM --platform=$BUILDPLATFORM rust:latest

WORKDIR /usr/src/igraph-rs

COPY . .

RUN apt-get update \
    && apt-get install -y sudo libglpk-dev liblapack-dev cmake build-essential wget clang \
    && rustup component add rustfmt \
    && cargo install bindgen-cli

RUN rm -rf target \
    && cd .. \
    && mkdir igraph \
    && cd igraph \
    && wget https://github.com/igraph/igraph/releases/download/1.0.0/igraph-1.0.0.tar.gz --no-verbose \
    && tar -xf igraph-1.0.0.tar.gz \
    && cd igraph-1.0.0 \
    && mkdir build \
    && cd build \
    && cmake -DBUILD_SHARED_LIBS=ON .. \
    && cmake --build . \
    && sudo cmake --install . \
    && sudo ldconfig \
    && cd ../../../ \
    && rm -rf igraph \
    && cd igraph-rs \
    && make compile