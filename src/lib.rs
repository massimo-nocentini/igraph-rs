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

    #[test]
    fn test_igraph_first_example() {
        let num_vertices = 1000;
        let num_edges = 1000;

        unsafe {
            let mut graph = mem::zeroed::<igraph_t>();
            /* Initialize the library. */
            igraph_setup();
            /* Ensure identical results across runs. */
            igraph_rng_seed(igraph_rng_default(), 42);

            igraph_erdos_renyi_game_gnm(
                &mut graph,
                num_vertices,
                num_edges,
                false,
                IGRAPH_SIMPLE_SW,
                false,
            );

            let mut diameter = 0.0;

            igraph_diameter(
                &graph,
                std::ptr::null(),
                &mut diameter,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                IGRAPH_UNDIRECTED == 1,
                true,
            );

            let mut mean_degree = 0.0;
            igraph_mean_degree(&graph, &mut mean_degree, IGRAPH_LOOPS_SW == 1);

            // eprintln!("Diameter: {}, Mean degree: {}.", diameter, mean_degree);

            assert!(diameter == 23.0);
            assert!(mean_degree == 2.0);

            igraph_destroy(&mut graph);
        }
    }
}
