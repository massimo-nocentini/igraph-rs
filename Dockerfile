FROM --platform=$BUILDPLATFORM alpine:latest

LABEL org.opencontainers.image.description="Rust bindings for the igraph library"

ARG IGRAPH_VERSION=1.0.1

WORKDIR /usr/src/igraph-rs

# clang/clang-dev provide libclang, needed by bindgen to generate the bindings;
# libxml2-dev enables GraphML support. igraph's other dependencies (GMP, BLAS,
# LAPACK, ARPACK, GLPK, plfit) are vendored: the system ARPACK is not
# thread-safe, and the crate's tests run in parallel threads.
RUN apk add --no-cache cmake build-base wget clang clang-dev libxml2-dev flex bison rust cargo

# Build and install igraph as a shared library under /usr/local, where the
# build script looks for it. Recent compilers emit warnings that igraph would
# otherwise turn into errors, hence IGRAPH_WARNINGS_AS_ERRORS=OFF.
RUN cd /tmp \
    && wget https://github.com/igraph/igraph/releases/download/${IGRAPH_VERSION}/igraph-${IGRAPH_VERSION}.tar.gz --no-verbose \
    && tar -xf igraph-${IGRAPH_VERSION}.tar.gz \
    && cd igraph-${IGRAPH_VERSION} \
    && CC=clang CXX=clang++ cmake -S . -B build \
        -DBUILD_SHARED_LIBS=ON \
        -DIGRAPH_WARNINGS_AS_ERRORS=OFF \
        -DIGRAPH_ENABLE_TLS=ON \
        -DIGRAPH_USE_INTERNAL_ARPACK=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
    && cmake --build build --parallel \
    && cmake --install build \
    && cd /tmp \
    && rm -rf igraph-${IGRAPH_VERSION} igraph-${IGRAPH_VERSION}.tar.gz

COPY Cargo.toml Cargo.lock build.rs wrapper.h Makefile README.md ./
COPY src src
COPY tests tests
COPY examples examples

RUN make compile
