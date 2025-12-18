#![doc = include_str!("../README.md")]
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::mem;

include!("../bindings.rs");

impl Drop for igraph_t {
    fn drop(&mut self) {
        unsafe {
            igraph_destroy(self);
        }
    }
}

impl Drop for igraph_vector_int_t {
    fn drop(&mut self) {
        unsafe {
            igraph_vector_int_destroy(self);
        }
    }
}

impl igraph_vector_int_t {
    pub fn size(&self) -> usize {
        unsafe { igraph_vector_int_size(self) as usize }
    }

    pub fn with_capacity(size: usize) -> Self {
        unsafe {
            let mut vec = mem::zeroed::<igraph_vector_int_t>();
            igraph_vector_int_init(&mut vec, size as i64);
            vec
        }
    }

    pub fn set(&mut self, index: usize, value: i64) {
        unsafe {
            igraph_vector_int_set(self, index as i64, value);
        }
    }

    pub fn get(&self, index: usize) -> i64 {
        unsafe { igraph_vector_int_get(self, index as i64) }
    }
}

impl From<&[i64]> for igraph_vector_int_t {
    fn from(vec: &[i64]) -> Self {
        let mut igraph_vec = Self::with_capacity(vec.len());
        for (i, &value) in vec.iter().enumerate() {
            igraph_vec.set(i, value);
        }
        igraph_vec
    }
}

impl Into<Vec<i64>> for igraph_vector_int_t {
    fn into(self) -> Vec<i64> {
        let size = self.size();
        let mut vec = Vec::with_capacity(size);
        for i in 0..size {
            vec.push(self.get(i));
        }
        vec
    }
}

pub enum edge_type_sw_t {
    SIMPLE,
    LOOPS,
    MULTI,
}

impl igraph_rng_t {
    pub fn get_integer(&mut self, min: i64, max: i64) -> i64 {
        unsafe { igraph_rng_get_integer(self, min, max) }
    }

    pub fn get_unif(&mut self, min: f64, max: f64) -> f64 {
        unsafe { igraph_rng_get_unif(self, min, max) }
    }

    pub fn seed<'a>(seed: u64) -> Option<&'a mut igraph_rng_t> {
        unsafe {
            let rng = igraph_rng_default();
            igraph_rng_seed(rng, seed);
            rng.as_mut()
        }
    }
}

impl igraph_t {
    pub fn setup() {
        unsafe {
            igraph_setup();
        }
    }

    /// In the `G(n, m)` Erdős-Rényi model, a graph with `n` vertices and `m` edges is generated uniformly at random;
    /// for the sake of clarity, it binds the [igraph_erdos_renyi_game_gnm](https://igraph.org/c/html/latest/igraph-Games.html#igraph_erdos_renyi_game_gnm) function.
    pub fn erdos_renyi_game_gnm(
        num_vertices: usize,
        num_edges: usize,
        directed: bool,
        mode: edge_type_sw_t,
        edge_attr: bool,
    ) -> Self {
        unsafe {
            let mut graph = Self::new(0, directed);

            igraph_erdos_renyi_game_gnm(
                &mut graph,
                num_vertices as i64,
                num_edges as i64,
                directed,
                match mode {
                    edge_type_sw_t::SIMPLE => IGRAPH_SIMPLE_SW,
                    edge_type_sw_t::LOOPS => IGRAPH_LOOPS_SW,
                    edge_type_sw_t::MULTI => IGRAPH_MULTI_SW,
                },
                edge_attr,
            );
            graph
        }
    }

    pub fn new(num_vertices: usize, directed: bool) -> Self {
        unsafe {
            let mut graph = mem::zeroed::<igraph_t>();
            igraph_empty(&mut graph, num_vertices as i64, directed);
            graph
        }
    }

    pub fn num_vertices(&self) -> i64 {
        unsafe { igraph_vcount(self) }
    }

    pub fn num_edges(&self) -> i64 {
        unsafe { igraph_ecount(self) }
    }

    pub fn is_directed(&self) -> bool {
        unsafe { igraph_is_directed(self) }
    }

    pub fn add_vertices(&mut self, n: i64) {
        unsafe {
            igraph_add_vertices(self, n, std::ptr::null());
        }
    }

    pub fn add_edge(&mut self, from: i64, to: i64) {
        unsafe {
            igraph_add_edge(self, from, to);
        }
    }

    pub fn add_edges_from_slice(&mut self, edges_slice: &[(i64, i64)]) {
        unsafe {
            let mut edges = igraph_vector_int_t::with_capacity(edges_slice.len() * 2);
            let mut i = 0;
            for (from, to) in edges_slice.iter() {
                edges.set(i, *from);
                i += 1;
                edges.set(i, *to);
                i += 1;
            }
            igraph_add_edges(self, &edges, std::ptr::null());
        }
    }

    pub fn add_edges_from_vector(&mut self, edges: &igraph_vector_int_t) {
        unsafe {
            igraph_add_edges(self, edges, std::ptr::null());
        }
    }

    /// [Calculates the diameter of a graph (longest geodesic)](https://igraph.org/c/html/0.10.2/igraph-Structural.html#igraph_diameter):
    ///
    /// The diameter of a graph is the length of the longest shortest path it has.
    /// This function computes both the diameter, as well as the corresponding path.
    /// The diameter of the null graph is considered be infinity by convention.
    /// If the graph has no vertices, IGRAPH_NAN is returned.
    pub fn diameter(&self) -> f64 {
        let mut diameter = 0.0;
        unsafe {
            igraph_diameter(
                self,
                std::ptr::null(),
                &mut diameter,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                self.is_directed(),
                true,
            );
        }
        diameter
    }

    pub fn mean_degree(&self, loops: bool) -> f64 {
        let mut mean_degree = 0.0;
        unsafe {
            igraph_mean_degree(self, &mut mean_degree, loops);
        }
        mean_degree
    }

    pub fn community_multilevel(&self, resolution: f64) -> Vec<i64> {
        unsafe {
            let mut membership = igraph_vector_int_t::with_capacity(self.num_vertices() as usize);
            igraph_community_multilevel(
                self,
                std::ptr::null(),
                resolution,
                &mut membership,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            membership.into()
        }
    }
}

/// # Introduction
///
/// A simple test that creates a random graph and computes its diameter and mean degree.
/// It is a translation of the first example from the igraph C library documentation,
/// that can be found [in the first lesson](https://igraph.org/c/html/latest/igraph-Tutorial.html#tut-lesson-1).
///
/// # C code
///
/// ```c
/// #include <igraph.h>
///
/// int main(void) {
///    igraph_int_t num_vertices = 1000;
///    igraph_int_t num_edges = 1000;
///    igraph_real_t diameter, mean_degree;
///    igraph_t graph;
///
///    /* Initialize the library. */
///    igraph_setup();
///
///    /* Ensure identical results across runs. */
///    igraph_rng_seed(igraph_rng_default(), 42);
///
///    igraph_erdos_renyi_game_gnm(
///            &graph, num_vertices, num_edges,
///            IGRAPH_UNDIRECTED, IGRAPH_SIMPLE_SW, IGRAPH_EDGE_UNLABELED);
///
///    igraph_diameter(
///        &graph, /* weights = */ NULL,
///        &diameter,
///        /* from = */ NULL, /* to = */ NULL,
///        /* vertex_path = */ NULL, /* edge_path = */ NULL,
///        IGRAPH_UNDIRECTED, /* unconn= */ true);
///
///    igraph_mean_degree(&graph, &mean_degree, IGRAPH_LOOPS);
///    printf("Diameter of a random graph with average degree %g: %g\n",
///           mean_degree, diameter);
///
///    igraph_destroy(&graph);
///
///    return 0;
/// }
/// ```
fn example_1() {
    let graph = igraph_t::erdos_renyi_game_gnm(1000, 1000, false, edge_type_sw_t::SIMPLE, false);

    let diameter = graph.diameter();

    let mean_degree = graph.mean_degree(true);

    assert!(diameter == 23.0);
    assert!(mean_degree == 2.0);
}

fn example_2() {
    /*
    int main(void) {
        igraph_t graph;
        igraph_vector_int_t dimvector;
        igraph_vector_int_t edges;
        igraph_vector_bool_t periodic;
        igraph_real_t avg_path_len;

        /* Initialize the library. */
        igraph_setup();

        igraph_vector_int_init(&dimvector, 2);
        VECTOR(dimvector)[0] = 30;
        VECTOR(dimvector)[1] = 30;

        igraph_vector_bool_init(&periodic, 2);
        igraph_vector_bool_fill(&periodic, true);
        igraph_square_lattice(&graph, &dimvector, 0, IGRAPH_UNDIRECTED,
                              /* mutual= */ false, &periodic);

        igraph_average_path_length(&graph, NULL, &avg_path_len, NULL,
                                   IGRAPH_UNDIRECTED, /* unconn= */ true);
        printf("Average path length (lattice):            %g\n", (double) avg_path_len);

        /* Seed the RNG to ensure identical results across runs. */
        igraph_rng_seed(igraph_rng_default(), 42);

        igraph_vector_int_init(&edges, 20);
        for (igraph_int_t i = 0; i < igraph_vector_int_size(&edges); i++) {
            VECTOR(edges)[i] = RNG_INTEGER(0, igraph_vcount(&graph) - 1);
        }

        igraph_add_edges(&graph, &edges, NULL);
        igraph_average_path_length(&graph, NULL, &avg_path_len, NULL,
                                   IGRAPH_UNDIRECTED, /* unconn= */ true);
        printf("Average path length (randomized lattice): %g\n", (double) avg_path_len);

        igraph_vector_bool_destroy(&periodic);
        igraph_vector_int_destroy(&dimvector);
        igraph_vector_int_destroy(&edges);
        igraph_destroy(&graph);

        return 0;
    }
    */

    unsafe {
        let mut graph = mem::zeroed::<igraph_t>();
        igraph_setup();
        let mut dimvector = mem::zeroed::<igraph_vector_int_t>();
        let mut periodic = mem::zeroed::<igraph_vector_bool_t>();
        igraph_vector_int_init(&mut dimvector, 2);
        igraph_vector_int_set(&mut dimvector, 0, 30);
        igraph_vector_int_set(&mut dimvector, 1, 30);

        igraph_vector_bool_init(&mut periodic, 2);
        igraph_vector_bool_fill(&mut periodic, true);
        igraph_square_lattice(
            &mut graph,
            &dimvector,
            0,
            IGRAPH_UNDIRECTED == 1,
            false,
            &periodic,
        );

        let mut avg_path_len = 0.0;
        igraph_average_path_length(
            &graph,
            std::ptr::null(),
            &mut avg_path_len,
            std::ptr::null_mut(),
            IGRAPH_UNDIRECTED == 1,
            true,
        );
        println!("Average path length (lattice):            {}", avg_path_len);

        let rng = igraph_rng_default();
        igraph_rng_seed(rng, 42);

        let mut edges = mem::zeroed::<igraph_vector_int_t>();
        igraph_vector_int_init(&mut edges, 20);
        for i in 0..igraph_vector_int_size(&edges) {
            let rand_vertex = igraph_rng_get_integer(rng, 0, igraph_vcount(&graph) - 1);
            igraph_vector_int_set(&mut edges, i, rand_vertex);
        }

        igraph_add_edges(&mut graph, &edges, std::ptr::null());
        igraph_average_path_length(
            &graph,
            std::ptr::null(),
            &mut avg_path_len,
            std::ptr::null_mut(),
            IGRAPH_UNDIRECTED == 1,
            true,
        );
        println!("Average path length (randomized lattice): {}", avg_path_len);

        igraph_vector_bool_destroy(&mut periodic);
    }
}

/// In our next example we will calculate various centrality measures in a friendship graph.
/// The friendship graph is from the famous Zachary karate club study.
/// (Do a web search on "Zachary karate" if you want to know more about this.)
/// Centrality measures quantify how central is the position of individual vertices in the graph.
///
fn example_3() {
    /*
        int main(void) {
        igraph_t graph;
        igraph_vector_int_t result;
        igraph_vector_t result_real;
        igraph_int_t edges_array[] = {
            0,1, 0,2, 0,3, 0,4, 0,5, 0,6, 0,7, 0,8,
            0,10, 0,11, 0,12, 0,13, 0,17, 0,19, 0,21, 0,31,
            1, 2, 1, 3, 1, 7, 1,13, 1,17, 1,19, 1,21, 1,30,
            2, 3, 2, 7, 2,27, 2,28, 2,32, 2, 9, 2, 8, 2,13,
            3, 7, 3,12, 3,13, 4, 6, 4,10, 5, 6, 5,10, 5,16,
            6,16, 8,30, 8,32, 8,33, 9,33, 13,33, 14,32, 14,33,
            15,32, 15,33, 18,32, 18,33, 19,33, 20,32, 20,33,
            22,32, 22,33, 23,25, 23,27, 23,32, 23,33, 23,29,
            24,25, 24,27, 24,31, 25,31, 26,29, 26,33, 27,33,
            28,31, 28,33, 29,32, 29,33, 30,32, 30,33, 31,32,
            31,33, 32,33
        };
        igraph_vector_int_t edges =
            igraph_vector_int_view(edges_array, sizeof(edges_array) / sizeof(edges_array[0]));

        /* Initialize the library. */
        igraph_setup();

        igraph_create(&graph, &edges, 0, IGRAPH_UNDIRECTED);

        igraph_vector_int_init(&result, 0);
        igraph_vector_init(&result_real, 0);

        igraph_degree(&graph, &result, igraph_vss_all(), IGRAPH_ALL, IGRAPH_LOOPS);
        printf("Maximum degree is      %10" IGRAPH_PRId ", vertex %2" IGRAPH_PRId ".\n",
               igraph_vector_int_max(&result),
               igraph_vector_int_which_max(&result));

        igraph_closeness(&graph, &result_real, NULL, NULL, igraph_vss_all(),
                         IGRAPH_ALL, /* weights= */ NULL, /* normalized= */ false);
        printf("Maximum closeness is   %10g, vertex %2" IGRAPH_PRId ".\n",
               (double) igraph_vector_max(&result_real),
               igraph_vector_which_max(&result_real));

        igraph_betweenness(&graph, /* weights= */ NULL, &result_real, igraph_vss_all(),
                           IGRAPH_UNDIRECTED, /* normalized= */ false);
        printf("Maximum betweenness is %10g, vertex %2" IGRAPH_PRId ".\n",
               (double) igraph_vector_max(&result_real),
               igraph_vector_which_max(&result_real));

        igraph_vector_int_destroy(&result);
        igraph_vector_destroy(&result_real);
        igraph_destroy(&graph);

        return 0;
    }
         */

    unsafe {
        let mut graph = mem::zeroed::<igraph_t>();

        let edges_array: [i64; 156] = [
            0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 10, 0, 11, 0, 12, 0, 13, 0, 17, 0,
            19, 0, 21, 0, 31, 1, 2, 1, 3, 1, 7, 1, 13, 1, 17, 1, 19, 1, 21, 1, 30, 2, 3, 2, 7, 2,
            27, 2, 28, 2, 32, 2, 9, 2, 8, 2, 13, 3, 7, 3, 12, 3, 13, 4, 6, 4, 10, 5, 6, 5, 10, 5,
            16, 6, 16, 8, 30, 8, 32, 8, 33, 9, 33, 13, 33, 14, 32, 14, 33, 15, 32, 15, 33, 18, 32,
            18, 33, 19, 33, 20, 32, 20, 33, 22, 32, 22, 33, 23, 25, 23, 27, 23, 32, 23, 33, 23, 29,
            24, 25, 24, 27, 24, 31, 25, 31, 26, 29, 26, 33, 27, 33, 28, 31, 28, 33, 29, 32, 29, 33,
            30, 32, 30, 33, 31, 32, 31, 33, 32, 33,
        ];
        let edges = igraph_vector_int_view(edges_array.as_ptr(), edges_array.len() as i64);

        igraph_setup();

        igraph_create(&mut graph, &edges, 0, IGRAPH_UNDIRECTED == 1);

        let mut result = mem::zeroed::<igraph_vector_int_t>();
        let mut result_real = mem::zeroed::<igraph_vector_t>();
        igraph_vector_int_init(&mut result, 0);
        igraph_vector_init(&mut result_real, 0);

        igraph_degree(
            &graph,
            &mut result,
            igraph_vss_all(),
            igraph_neimode_t_IGRAPH_ALL,
            IGRAPH_LOOPS_SW,
        );
        println!(
            "Maximum degree is      {:10}, vertex {:2}.",
            igraph_vector_int_max(&result),
            igraph_vector_int_which_max(&result),
        );

        igraph_closeness(
            &graph,
            &mut result_real,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            igraph_vss_all(),
            igraph_neimode_t_IGRAPH_ALL,
            std::ptr::null(),
            false,
        );
        println!(
            "Maximum closeness is   {:10}, vertex {:2}.",
            igraph_vector_max(&result_real),
            igraph_vector_which_max(&result_real)
        );

        igraph_betweenness(
            &graph,
            std::ptr::null(),
            &mut result_real,
            igraph_vss_all(),
            IGRAPH_UNDIRECTED == 1,
            false,
        );
        println!(
            "Maximum betweenness is {:10}, vertex {:2}.",
            igraph_vector_max(&result_real),
            igraph_vector_which_max(&result_real)
        );

        igraph_vector_destroy(&mut result_real);
    }
}

#[cfg(test)]
mod tests {

    use super::*;

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
    fn test_igraph_tutorial() {
        igraph_t::setup();
        let _rng = igraph_rng_t::seed(42).unwrap();
        example_1();
        example_2();
        // example_3();
    }
}
