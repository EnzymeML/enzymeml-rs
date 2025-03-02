//! Tests for the optimization module.
//!
//! This module contains tests for the different optimization algorithms:
//! - BFGS (Broyden-Fletcher-Goldfarb-Shanno)
//! - L-BFGS (Limited-memory BFGS)
//! - PSO (Particle Swarm Optimization)
//! - EGO (Efficient Global Optimization)
//!
//! Each test verifies that the optimizer can find the correct parameters
//! for a simple enzyme kinetics model with known parameters.

#[cfg(test)]
mod test_optim {
    use approx::assert_relative_eq;
    use enzymeml::{
        optim::{
            BFGSBuilder, EGOBuilder, LBFGSBuilder, Optimizer, PSOBuilder, ProblemBuilder,
            Transformation,
        },
        prelude::*,
    };
    use ndarray::Array1;
    use std::path::PathBuf;

    fn get_doc() -> EnzymeMLDocument {
        let path = PathBuf::from("tests/data/enzmldoc.json");
        let doc = load_enzmldoc(&path).unwrap();
        doc
    }

    #[test]
    fn test_bfgs() {
        // ARRANGE
        let doc = get_doc();
        let problem = ProblemBuilder::new(&doc)
            .dt(10.0)
            .transform(Transformation::Log("k_cat".into()))
            .transform(Transformation::NegExp("k_ie".into()))
            .transform(Transformation::MultScale("K_M".into(), 10.0))
            .build()
            .expect("Failed to build problem");

        // ACT
        let bfgs = BFGSBuilder::default()
            .linesearch(1e-4, 0.9)
            .max_iters(20)
            .target_cost(1e-6)
            .build();

        let inits = Array1::from_vec(vec![18.0, 2.0, 7.0]);
        let res = bfgs
            .optimize(&problem, Some(inits))
            .expect("Failed to optimize");

        // ASSERT
        let k_m = res[0];
        let k_cat = res[1];
        let k_ie = res[2];
        assert_relative_eq!(k_m, 8.0, epsilon = 2.0);
        assert_relative_eq!(k_cat, 2.0, epsilon = 2.0);
        assert_relative_eq!(k_ie, 6.01, epsilon = 4.0);
    }

    #[test]
    fn test_lbfgs() {
        // ARRANGE
        let doc = get_doc();
        let problem = ProblemBuilder::new(&doc)
            .dt(10.0)
            .transform(Transformation::Log("k_cat".into()))
            .transform(Transformation::NegExp("k_ie".into()))
            .transform(Transformation::MultScale("K_M".into(), 10.0))
            .build()
            .expect("Failed to build problem");

        // ACT
        let lbfgs = LBFGSBuilder::default()
            .linesearch(1e-4, 0.9)
            .max_iters(20)
            .target_cost(1e-6)
            .build();

        let inits = Array1::from_vec(vec![18.0, 2.0, 7.0]);
        let res = lbfgs
            .optimize(&problem, Some(inits))
            .expect("Failed to optimize");

        // ASSERT
        let k_m = res[0];
        let k_cat = res[1];
        let k_ie = res[2];
        assert_relative_eq!(k_m, 8.0, epsilon = 2.0);
        assert_relative_eq!(k_cat, 2.0, epsilon = 2.0);
        assert_relative_eq!(k_ie, 6.01, epsilon = 4.0);
    }

    #[test]
    fn test_pso() {
        // ARRANGE
        let doc = get_doc();
        let problem = ProblemBuilder::new(&doc)
            .dt(10.0)
            .build()
            .expect("Failed to build problem");

        // ACT
        let pso = PSOBuilder::default()
            .pop_size(100)
            .max_iters(20)
            .bound("K_M", 1e-6, 120.0)
            .bound("k_cat", 1e-6, 1.0)
            .bound("k_ie", 1e-6, 0.005)
            .build();

        let res = pso
            .optimize(&problem, None::<Array1<f64>>)
            .expect("Failed to optimize");

        // ASSERT
        let k_m = res[0];
        let k_cat = res[1];
        let k_ie = res[2];
        assert_relative_eq!(k_m, 82.0, epsilon = 5.0);
        assert_relative_eq!(k_cat, 0.85, epsilon = 0.1);
        assert_relative_eq!(k_ie, 0.001, epsilon = 0.01);
    }

    #[test]
    fn test_ego() {
        // ARRANGE
        let doc = get_doc();
        let problem = ProblemBuilder::new(&doc).dt(10.0).build().unwrap();

        // ACT
        let res = EGOBuilder::default()
            .max_iters(50)
            .bound("K_m", 1e-6, 120.0)
            .bound("k_cat", 1e-6, 1.0)
            .bound("k_ie", 1e-6, 0.005)
            .build()
            .optimize(&problem, None::<Array1<f64>>)
            .unwrap();

        // ASSERT
        let k_m = res[0];
        let k_cat = res[1];
        let k_ie = res[2];
        assert_relative_eq!(k_m, 82.0, epsilon = 5.0);
        assert_relative_eq!(k_cat, 0.85, epsilon = 0.1);
        assert_relative_eq!(k_ie, 0.001, epsilon = 0.01);
    }
}
