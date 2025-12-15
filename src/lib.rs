#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!("../bindings.rs");

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn test_igraph_version() {
        let major = IGRAPH_VERSION_MAJOR;
        let minor = IGRAPH_VERSION_MINOR;
        let patch = IGRAPH_VERSION_PATCH;
        assert!(major == 1);
        assert!(minor == 0);
        assert!(patch == 0);
    }
}
