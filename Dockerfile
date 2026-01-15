
FROM --platform=$BUILDPLATFORM rust:latest

LABEL org.opencontainers.image.description="Rust bindings for the igraph library"

WORKDIR /usr/src/igraph-rs

COPY src src
COPY Cargo.toml .
COPY Cargo.lock .
COPY build.rs .
COPY bindings.rs .

RUN apt-get update \
    && apt-get install -y sudo libglpk-dev liblapack-dev cmake build-essential wget clang flex bison libc++-dev \
    && rustup component add rustfmt \
    && cargo install bindgen-cli

RUN export IGRAPH_VERSION="1.0.1" \
    && cd .. \
    && mkdir igraph \
    && cd igraph \
    && wget https://github.com/igraph/igraph/releases/download/${IGRAPH_VERSION}/igraph-${IGRAPH_VERSION}.tar.gz --no-verbose \
    && tar -xf igraph-${IGRAPH_VERSION}.tar.gz \
    && cd igraph-${IGRAPH_VERSION} \
    && mkdir build \
    && cd build \
    && export CXX=clang++ \
    && export CC=clang \
    && cmake -DBUILD_SHARED_LIBS=ON .. \
    && cmake --build . \
    && sudo cmake --install . \
    && sudo ldconfig \
    && cd ../../../ \
    && rm -rf igraph \
    && cd igraph-rs \
    && make compile