fn main() {
    // According to https://doc.rust-lang.org/cargo/reference/build-scripts.html#rustc-link-lib:
    println!("cargo:rustc-link-search=/usr/local/lib");
    println!("cargo:rustc-link-lib=igraph");
    println!("cargo:rustc-link-lib=c++");
}
