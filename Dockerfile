
FROM --platform=$BUILDPLATFORM alpine:latest

LABEL org.opencontainers.image.description="Rust bindings for the igraph library"

WORKDIR /usr/src/igraph-rs

RUN apk add glpk-dev lapack-dev cmake build-base wget clang flex bison libc++-dev rust cargo

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
    && cmake --install . \
    && cd ../../../ \
    && rm -rf igraph

COPY src src
COPY Cargo.toml .
COPY Cargo.lock .
COPY build.rs .
COPY bindings.rs .
COPY Makefile .
COPY README.md .

RUN make compile