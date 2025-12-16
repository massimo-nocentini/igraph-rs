
bindgen:
	bindgen /usr/local/include/igraph/igraph.h -o bindings.rs

compile:
	cargo build --release
	cargo test --release -- --nocapture

doc:
	cargo doc --document-private-items --release
	cp -r target/doc ./docs

docker-build:
	docker build -t ghcr.io/massimo-nocentini/igraph-rs:master .

docker-run:
	docker run -it --rm ghcr.io/massimo-nocentini/igraph-rs:master