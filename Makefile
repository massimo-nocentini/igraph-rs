# The build script finds igraph 1.0.1 in /usr/local or ~/.local by itself.
# For other locations pass the installation prefix, e.g.
# `make test IGRAPH_DIR=/opt/igraph` (or IGRAPH_INCLUDE_DIR / IGRAPH_LIB_DIR).
ifdef IGRAPH_DIR
export IGRAPH_DIR
endif

.PHONY: compile test doc docker-build docker-run

compile:
	cargo build --release
	cargo test --release -- --nocapture

test:
	cargo test

doc:
	cargo doc --no-deps --release
	rm -rf docs && cp -r target/doc docs && echo '<meta http-equiv="refresh" content="0; url=igraph/index.html">' > docs/index.html

docker-build:
	docker build -t ghcr.io/massimo-nocentini/igraph-rs:master . --no-cache

docker-run:
	docker run -it --rm ghcr.io/massimo-nocentini/igraph-rs:master
