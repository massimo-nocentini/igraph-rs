# Introduction

Low-level, unsafe Rust bindings to the C igraph library (generated via bindgen).
This crate exposes the raw FFI types and functions from igraph through the
included `bindings.rs` and provides example translations of the igraph C
tutorial demonstrating common workflows:
- [lesson 1](fn.example_1.html): create an Erdős–Rényi random graph, compute diameter and mean degree.
- [lesson 2](fn.example_2.html): build a square lattice, measure average path length, randomize edges.
- [lesson 3](fn.example_3.html): construct the Zachary karate club friendship graph and compute degree,
  closeness and betweenness centralities.

## Safety and usage notes
- All examples use unsafe code and direct FFI calls. Callers must respect igraph's
  initialization and destruction APIs (e.g. `igraph_setup()` and `igraph_destroy()`),
  and must correctly initialize and destroy igraph vector types.
- Randomness can be made deterministic by seeding the igraph RNG via
  `igraph_rng_seed(igraph_rng_default(), seed)`.
- Many constants are C macros exposed by the bindings; their values may be compared
  as integers (e.g. `IGRAPH_UNDIRECTED == 1`) when used as boolean flags.
- Memory management follows the C library's expectations: vectors and graph objects
  created via igraph APIs must be explicitly destroyed with the corresponding
  `igraph_*_destroy` functions to avoid leaks.

## Tests
- `test_igraph_version` asserts compile-time/runtime version constants from the
  bindings (major/minor/patch).
- `test_igraph_tutorial` runs the three tutorial examples to validate the bindings
  and example translations.

## Intended audience
- Users who need direct access to igraph's C API from Rust for prototyping,
  testing, or building higher-level safe wrappers. Because the bindings are low-level,
  higher-level ergonomic abstractions are recommended for production code.

See the upstream igraph C documentation and the original tutorial examples for
algorithmic details and semantics of the used functions.

## Publications

[Csardi, G., & Nepusz, T. (2006). The igraph software package for complex network research. InterJournal, Complex Systems, 1695.](https://www.semanticscholar.org/paper/The-igraph-software-package-for-complex-network-Cs%C3%A1rdi-Nepusz/1d2744b83519657f5f2610698a8ddd177ced4f5c)

# Installation

We use the [`bindgen`](https://rust-lang.github.io/rust-bindgen/tutorial-1.html) crate to build Rust bindings over the 
[`igraph` C library](https://github.com/igraph/igraph).

The latest version of the original C library can be fetched on 
[Github](https://github.com/igraph/igraph/releases/download/1.0.0/igraph-1.0.0.tar.gz) and the installation
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
provided that the main tarball as been extracted and that the following dependencies (on MacOS, but similar on Linux distros)
```sh
brew install gmp lapack libxml2 glpk arpack blas suitesparse
```
are installed as well.
