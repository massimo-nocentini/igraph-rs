fn main() {
    // According to https://doc.rust-lang.org/cargo/reference/build-scripts.html#rustc-link-lib:
    println!("cargo:rustc-link-lib=dylib=igraph");
    // println!("cargo:rustc-link-lib=dylib=stdc++");
    // println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-search=native=/usr/local/lib");
    // println!("cargo:rustc-link-search=native=/opt/homebrew/opt/gcc/lib/gcc/current");
}
