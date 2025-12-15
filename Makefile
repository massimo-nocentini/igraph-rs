
bindgen:
	bindgen /usr/local/include/igraph/igraph.h -o bindings.rs

test:
	cargo test --release