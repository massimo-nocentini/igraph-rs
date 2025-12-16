# igraph-rs

Rust bindings for the igraph C library for creating and manipulating graphs.

We are using the [`bindgen`](https://rust-lang.github.io/rust-bindgen/tutorial-1.html) crate to build Rust bindings over the [`igraph` C library](https://github.com/igraph/igraph), after the publication of [Csardi, G., & Nepusz, T. (2006). The igraph software package for complex network research. InterJournal, Complex Systems, 1695.](https://www.semanticscholar.org/paper/The-igraph-software-package-for-complex-network-Cs%C3%A1rdi-Nepusz/1d2744b83519657f5f2610698a8ddd177ced4f5c)

The latest version of the original C library can be fetched at https://github.com/igraph/igraph/releases/download/1.0.0/igraph-1.0.0.tar.gz and the installation
proceeds according to the [tutorial](https://igraph.org/c/):
```sh
wget https://github.com/igraph/igraph/releases/download/1.0.0/igraph-1.0.0.tar.gz
tar xf igraph-1.0.0.tar.gz
cd igraph-1.0.0
mkdir build && cd build
cmake .. # -DBUILD_SHARED_LIBS=ON for building shared libraries.
cmake --build .
sudo cmake --install .
```
provided that the main tarball as been extracted and that the following dependencies:
```sh
brew install gmp lapack libxml2 glpk arpack blas suitesparse
```
are installed as well.