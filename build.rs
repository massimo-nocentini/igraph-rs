//! Build script: locates the igraph C library, links against it and generates
//! the raw FFI bindings with [`bindgen`](https://rust-lang.github.io/rust-bindgen/)
//! for the host platform.
//!
//! The location of igraph is discovered through the following environment
//! variables (checked in order of precedence):
//!
//! - `IGRAPH_INCLUDE_DIR`: directory containing `igraph.h`;
//! - `IGRAPH_LIB_DIR`: directory containing `libigraph.{so,dylib,a}`;
//! - `IGRAPH_DIR`: an installation prefix, from which `include/igraph` and
//!   `lib` are derived when the previous two variables are not set.
//!
//! When none of them is given, the first of `/usr/local` (where
//! `cmake --install` puts igraph by default) and `$HOME/.local` containing
//! `include/igraph/igraph.h` is used.

use bindgen::{
    FieldVisibilityKind,
    callbacks::{FieldInfo, ParseCallbacks},
};
use std::{env, path::PathBuf};

/// Plain-data structs (numbers only, no pointers nor sizes that igraph trusts)
/// whose fields stay `pub`: users may freely read and tweak them.
const PLAIN_DATA_STRUCTS: &[&str] = &["igraph_layout_drl_options_t", "igraph_maxflow_stats_t"];

#[derive(Debug)]
struct PublicPlainData;

impl ParseCallbacks for PublicPlainData {
    fn field_visibility(&self, info: FieldInfo<'_>) -> Option<FieldVisibilityKind> {
        PLAIN_DATA_STRUCTS
            .contains(&info.type_name)
            .then_some(FieldVisibilityKind::Public)
    }
}

fn main() {
    for var in ["IGRAPH_DIR", "IGRAPH_INCLUDE_DIR", "IGRAPH_LIB_DIR"] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    println!("cargo:rerun-if-changed=wrapper.h");

    let prefix = env::var("IGRAPH_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_prefix());
    let include_dir = env::var("IGRAPH_INCLUDE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| prefix.join("include").join("igraph"));
    let lib_dir = env::var("IGRAPH_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| prefix.join("lib"));

    // According to https://doc.rust-lang.org/cargo/reference/build-scripts.html#rustc-link-lib:
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=igraph");
    // Embed the library directory so that tests and examples run without
    // having to tweak `LD_LIBRARY_PATH`/`DYLD_LIBRARY_PATH`.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    // Emit DT_RPATH rather than DT_RUNPATH on ELF platforms, so that the
    // embedded directory wins over `LD_LIBRARY_PATH`: the library loaded at run
    // time is then the one whose headers generated the bindings.
    if env::var("CARGO_CFG_TARGET_OS").is_ok_and(|os| os != "macos" && os != "ios") {
        println!("cargo:rustc-link-arg=-Wl,--disable-new-dtags");
    }

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", include_dir.display()))
        .allowlist_function("igraph_.*|IGRAPH_.*")
        .allowlist_type("igraph_.*")
        .allowlist_var("IGRAPH_.*|igraph_.*")
        // A few libc functions needed to hand `FILE *` streams to igraph's I/O.
        .allowlist_function(
            "fopen|fclose|fflush|ferror|fmemopen|open_memstream|free|tmpfile|rewind",
        )
        .derive_default(false)
        .derive_debug(true)
        // Rusty wrappers implement `Drop` on the owning igraph types, which is
        // incompatible with `Copy`: never derive it.
        .derive_copy(false)
        // Struct fields hold the invariants of the owning wrappers (e.g. a
        // vector's storage pointers): safe code outside the crate must not be
        // able to change them, hence crate-visible only.
        .default_visibility(FieldVisibilityKind::PublicCrate)
        .layout_tests(false)
        .generate_comments(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .parse_callbacks(Box::new(PublicPlainData))
        .generate()
        .expect("Unable to generate the igraph bindings: is IGRAPH_DIR set correctly?");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

/// The first of `/usr/local` and `$HOME/.local` that contains igraph's
/// headers, falling back to `/usr/local`.
fn default_prefix() -> PathBuf {
    let mut candidates = vec![PathBuf::from("/usr/local")];
    if let Some(home) = env::var_os("HOME") {
        candidates.push(PathBuf::from(home).join(".local"));
    }
    candidates
        .iter()
        .find(|p| p.join("include/igraph/igraph.h").exists())
        .cloned()
        .unwrap_or_else(|| candidates.swap_remove(0))
}
