#[cfg(test)]
mod test_simulation {
    use std::collections::HashMap;

    use approx::assert_relative_eq;
    use enzymeml::io::load_enzmldoc;
    use enzymeml::prelude::{
        EnzymeMLDocument, EnzymeMLDocumentBuilder, EnzymeMLDocumentBuilderError, EquationBuilder,
        EquationType, MatrixResult, Mode, ODESystem, ParameterBuilder, PlotConfig,
        SimulationResult, SimulationSetupBuilder, StepperOutput,
    };
    use ndarray::Axis;
    use peroxide::fuga::{BasicODESolver, ODEProblem, ODESolver, RK5};

    /// Tests the basic simulation of a Michaelis-Menten enzyme kinetics system.
    /// Compares results between a reference implementation and an EnzymeML-based system.
    ///
    /// The test:
    /// 1. Creates a reference Michaelis-Menten system with KM = 100.0, Vmax = 10.0
    /// 2. Creates an equivalent EnzymeML-based system
    /// 3. Simulates both systems with identical initial conditions
    /// 4. Compares the results to ensure they match within tolerance
    #[test]
    fn test_simulation() {
        // ARRANGE
        // Configure simulation parameters (time range, step size, tolerances)
        let setup = SimulationSetupBuilder::default()
            .t0(0.0) // Start time
            .t1(10.0) // End time
            .dt(1.0) // Time step
            .rtol(1e-6) // Relative tolerance
            .atol(1e-6) // Absolute tolerance
            .build()
            .unwrap();

        // Define initial substrate concentration
        let initial_conditions = HashMap::from([("substrate".to_string(), 100.0)]);

        // Create and simulate the reference Michaelis-Menten system
        // Parameters: KM = 100.0, Vmax = 10.0
        let menten_system = MentenSystem::new(100.0, 10.0);
        let menten_result = menten_system.integrate(
            *initial_conditions.get("substrate").unwrap(),
            setup.t0,
            setup.t1,
            setup.dt,
        );

        // ACT
        // Create and simulate the EnzymeML-based system
        let enzmldoc = create_enzmldoc().unwrap();
        let system: ODESystem = enzmldoc.try_into().unwrap();
        let result = system.integrate::<SimulationResult>(
            &setup,
            &initial_conditions,
            None,
            None,
            RK5,
            Some(Mode::Regular),
        );

        // ASSERT
        let expected_result = menten_result.iter().map(|row| row[0]).collect::<Vec<_>>();

        if let Ok(SimulationResult {
            species,
            parameter_sensitivities: _,
            time: _,
            assignments: _,
        }) = result
        {
            let substrate = species.get("substrate").expect("Species are not present");

            for (expected, actual) in expected_result.iter().zip(substrate.iter()) {
                assert_relative_eq!(*expected, *actual, epsilon = 1e-4);
            }
        } else {
            panic!("Result is not a MatrixResult");
        }
    }

    /// Tests bulk simulation capabilities by running multiple identical simulations
    /// in parallel and comparing results to a reference implementation.
    ///
    /// The test:
    /// 1. Sets up multiple identical simulation configurations
    /// 2. Runs parallel simulations using the bulk_integrate method
    /// 3. Verifies each simulation result matches the reference implementation
    #[test]
    fn test_bulk_simulation() {
        // ARRANGE
        // Configure simulation parameters (time range, step size, tolerances)
        let setup = SimulationSetupBuilder::default()
            .t0(0.0) // Start time
            .t1(10.0) // End time
            .dt(1.0) // Time step
            .rtol(1e-6) // Relative tolerance
            .atol(1e-6) // Absolute tolerance
            .build()
            .unwrap();

        // Define initial substrate concentration
        let initial_conditions = HashMap::from([("substrate".to_string(), 100.0)]);

        // Create and simulate the reference Michaelis-Menten system
        // Parameters: KM = 100.0, Vmax = 10.0
        let menten_system = MentenSystem::new(100.0, 10.0);
        let menten_result = menten_system.integrate(
            *initial_conditions.get("substrate").unwrap(),
            setup.t0,
            setup.t1,
            setup.dt,
        );

        // Create a vector of initial conditions
        let initial_conditions = vec![
            initial_conditions.clone(),
            initial_conditions.clone(),
            initial_conditions.clone(),
            initial_conditions.clone(),
            initial_conditions.clone(),
            initial_conditions.clone(),
            initial_conditions.clone(),
        ];

        let setups = vec![
            setup.clone(),
            setup.clone(),
            setup.clone(),
            setup.clone(),
            setup.clone(),
            setup.clone(),
            setup.clone(),
        ];

        // ACT
        // Create and simulate the EnzymeML-based system
        let enzmldoc = create_enzmldoc().unwrap();
        let system: ODESystem = enzmldoc.try_into().unwrap();
        let results = system
            .bulk_integrate::<SimulationResult>(
                &setups,
                &initial_conditions,
                None,
                None,
                RK5,
                Some(Mode::Regular),
            )
            .expect("Simulation failed");

        // ASSERT
        let expected_result = menten_result.iter().map(|row| row[0]).collect::<Vec<_>>();

        for result in results {
            let substrate = result
                .species
                .get("substrate")
                .expect("Species are not present");

            for (expected, actual) in expected_result.iter().zip(substrate.iter()) {
                assert_relative_eq!(*expected, *actual, epsilon = 1e-4);
            }
        }
    }

    /// Tests the sensitivity analysis functionality by comparing parameter sensitivities
    /// between a reference implementation and an EnzymeML-based system.
    ///
    /// Specifically checks sensitivities with respect to:
    /// - KM (Michaelis constant)
    /// - Vmax (maximum reaction velocity)
    ///
    /// The sensitivities indicate how changes in these parameters affect the substrate concentration.
    #[test]
    fn test_sensitivity_analysis() {
        // This test compares sensitivity analysis results between:
        // 1. A manually implemented Michaelis-Menten system (MentenSystem)
        // 2. An automatically generated system from EnzymeML document

        // ARRANGE
        // Configure simulation parameters (time range, step size, tolerances)
        let setup = SimulationSetupBuilder::default()
            .t0(0.0) // Start time
            .t1(10.0) // End time
            .dt(1.0) // Time step
            .rtol(1e-6) // Relative tolerance
            .atol(1e-6) // Absolute tolerance
            .build()
            .unwrap();

        // Define initial substrate concentration
        let initial_conditions = HashMap::from([("substrate".to_string(), 100.0)]);

        // Create and simulate the reference Michaelis-Menten system
        // Parameters: KM = 100.0, Vmax = 10.0
        let menten_system = MentenSystem::new(100.0, 10.0);
        let menten_result = menten_system.integrate(
            *initial_conditions.get("substrate").unwrap(),
            setup.t0,
            setup.t1,
            setup.dt,
        );

        // ACT
        // Create and simulate the EnzymeML-based system
        let enzmldoc = create_enzmldoc().unwrap();
        let system: ODESystem = enzmldoc.try_into().unwrap();
        let result = system.integrate::<MatrixResult>(
            &setup,
            &initial_conditions,
            None,
            None,
            RK5,
            Some(Mode::Sensitivity),
        );

        // ASSERT
        // Compare sensitivity analysis results between both implementations
        match result {
            Ok(MatrixResult {
                species: _,
                parameter_sensitivities: Some(parameter_sensitivities),
                times: _,
                assignments: _,
            }) => {
                // Iterate over each timepoint and compare sensitivities
                for (subview, menten_value) in parameter_sensitivities
                    .axis_iter(Axis(0))
                    .zip(menten_result)
                {
                    let enzmldoc_km = subview.get((0, 0)).unwrap(); // Sensitivity w.r.t. KM
                    let enzmldoc_vmax = subview.get((0, 1)).unwrap(); // Sensitivity w.r.t. Vmax
                    let menten_km = menten_value[1];
                    let menten_vmax = menten_value[2];

                    // Verify that sensitivities match within tolerance
                    assert_relative_eq!(*enzmldoc_km, menten_km, epsilon = 1e-10);
                    assert_relative_eq!(*enzmldoc_vmax, menten_vmax, epsilon = 1e-10);
                }
            }
            Ok(MatrixResult {
                parameter_sensitivities: None,
                ..
            }) => {
                panic!("Expected parameter sensitivities, but none were computed");
            }
            Err(e) => {
                panic!("Simulation failed: {e:?}");
            }
        }
    }

    /// Tests the assignment functionality in the EnzymeML system, which allows for
    /// derived variables that depend on other system variables.
    ///
    /// This test specifically verifies:
    /// 1. Initial assignments are correctly evaluated
    /// 2. Dynamic assignments (product concentration) are correctly calculated
    /// 3. Mass conservation (substrate + product = constant) is maintained
    #[test]
    fn test_assignments() {
        // ARRANGE
        // Configure simulation parameters (time range, step size, tolerances)
        let setup = SimulationSetupBuilder::default()
            .t0(0.0) // Start time
            .t1(10.0) // End time
            .dt(1.0) // Time step
            .rtol(1e-6) // Relative tolerance
            .atol(1e-6) // Absolute tolerance
            .build()
            .unwrap();

        // Define initial substrate concentration
        let initial_conditions = HashMap::from([
            ("substrate".to_string(), 100.0),
            ("product".to_string(), 0.0),
        ]);

        // Create and simulate the reference Michaelis-Menten system
        // Parameters: KM = 100.0, Vmax = 10.0
        let menten_system = MentenSystem::new(100.0, 10.0);
        let menten_result = menten_system.integrate(
            *initial_conditions.get("substrate").unwrap(),
            setup.t0,
            setup.t1,
            setup.dt,
        );

        // ACT
        // Create and simulate the EnzymeML-based system
        let mut enzmldoc = create_enzmldoc().unwrap();
        add_assignments(&mut enzmldoc);
        let system: ODESystem = enzmldoc.try_into().unwrap();
        let result = system.integrate::<SimulationResult>(
            &setup,
            &initial_conditions,
            None,
            None,
            RK5,
            Some(Mode::Regular),
        );

        // ASSERT
        let expected_result = menten_result.iter().map(|row| row[0]).collect::<Vec<_>>();

        if let Ok(SimulationResult {
            species,
            parameter_sensitivities: _,
            time: _,
            assignments,
        }) = result
        {
            let assignments = assignments.expect("Assignments are not present");
            let product = assignments.get("product").expect("Product is not present");
            let substrate = species.get("substrate").expect("Species are not present");
            let substrate_initial = substrate.first().expect("Substrate is empty");

            // Transform the expected result to a vector of f64
            let expected_product = expected_result
                .iter()
                .map(|row| substrate_initial - row)
                .collect::<Vec<_>>();

            for (expected, actual) in expected_product.iter().zip(product.iter()) {
                assert_relative_eq!(*expected, *actual, epsilon = 1e-4);
            }
        } else {
            panic!("Result is not a MatrixResult");
        }
    }

    #[test]
    fn test_plot() {
        let enzmldoc = create_enzmldoc().unwrap();
        let system: ODESystem = enzmldoc.try_into().unwrap();
        let initial_conditions = HashMap::from([("substrate".to_string(), 100.0)]);
        let setup = SimulationSetupBuilder::default()
            .t0(0.0)
            .t1(10.0)
            .dt(1.0)
            .build()
            .unwrap();
        let result = system
            .integrate::<SimulationResult>(
                &setup,
                &initial_conditions,
                None,
                None,
                RK5,
                Some(Mode::Regular),
            )
            .expect("Simulation failed");

        let plot = result.plot(PlotConfig::default(), false);
        plot.to_html();
    }

    /// Tests sensitivity analysis using the enzmldoc.json test data to debug zero sensitivity issue.
    /// This test demonstrates the problem where sensitivity analysis returns zeros for reaction-based systems.
    #[test]
    fn test_sensitivity_analysis_enzmldoc_json() {
        // Load the test data from enzmldoc.json - this contains reactions, not direct ODEs
        let doc = load_enzmldoc("tests/data/enzmldoc.json").expect("Failed to load enzmldoc.json");

        // Create the ODE system from the document
        let system: ODESystem = (&doc).try_into().expect("Failed to create ODE system");

        // Print some debug info to understand the system
        println!("System has {} equations", system.num_equations());
        println!("System has {} parameters", system.num_parameters());
        println!("System parameters: {:?}", system.get_sorted_params());
        println!("System species: {:?}", system.get_sorted_species());

        // Configure simulation parameters using the first measurement's initial conditions
        let setup = SimulationSetupBuilder::default()
            .t0(0.0)
            .t1(100.0) // Use a shorter time span for faster testing
            .dt(10.0) // Larger time step for faster testing
            .build()
            .unwrap();

        // Extract initial conditions from the first measurement
        let first_measurement = &doc.measurements[0];
        let mut initial_conditions = HashMap::new();

        // Get initial conditions for all species from the measurement
        for species_data in &first_measurement.species_data {
            if let Some(initial_value) = species_data.initial {
                initial_conditions.insert(species_data.species_id.clone(), initial_value);
            } else {
                // Use a default value if initial is None
                initial_conditions.insert(species_data.species_id.clone(), 0.0);
            }
        }

        println!("Initial conditions: {initial_conditions:?}");

        // Run simulation with sensitivity analysis
        let result = system.integrate::<MatrixResult>(
            &setup,
            &initial_conditions,
            None,
            Some(&vec![10.0, 50.0, 100.0]), // Evaluate at specific time points
            peroxide::fuga::RK4,            // Use a simpler solver for debugging
            Some(Mode::Sensitivity),
        );

        // Check the result
        match result {
            Ok(MatrixResult {
                species,
                parameter_sensitivities: Some(parameter_sensitivities),
                times,
                assignments: _,
            }) => {
                println!("Species shape: {:?}", species.shape());
                println!(
                    "Parameter sensitivities shape: {:?}",
                    parameter_sensitivities.shape()
                );
                println!("Times shape: {:?}", times.shape());

                // Print actual sensitivity values to debug
                println!("Parameter sensitivities:");
                for (t_idx, time) in times.iter().enumerate() {
                    println!("Time {time}: ");
                    for species_idx in 0..parameter_sensitivities.shape()[1] {
                        print!("  Species {species_idx}: [");
                        for param_idx in 0..parameter_sensitivities.shape()[2] {
                            print!(
                                "{:.6}, ",
                                parameter_sensitivities[[t_idx, species_idx, param_idx]]
                            );
                        }
                        println!("]");
                    }
                }

                // Check if sensitivities are all zero (the bug we're investigating)
                let mut all_zero = true;
                for val in parameter_sensitivities.iter() {
                    if val.abs() > 1e-10 {
                        all_zero = false;
                        break;
                    }
                }

                if all_zero {
                    println!("ERROR: All sensitivities are zero! This indicates the bug.");
                } else {
                    println!("Sensitivities are non-zero, which is expected.");
                }

                // For now, we expect this test to fail until the bug is fixed
                assert!(!all_zero, "Sensitivities should not all be zero - this indicates a bug in the sensitivity calculation for reaction-based systems");
            }
            Ok(MatrixResult {
                parameter_sensitivities: None,
                ..
            }) => {
                panic!("Expected parameter sensitivities, but none were computed");
            }
            Err(e) => {
                println!("Simulation failed with error: {e:?}");

                // Check if this is the unimplemented error for stoichiometry-based sensitivity
                let error_string = format!("{e:?}");
                if error_string.contains("not supported yet")
                    || error_string.contains("unimplemented")
                {
                    println!("CONFIRMED: This is the stoichiometry sensitivity bug!");
                    panic!("Sensitivity analysis with stoichiometry matrix is not implemented yet - this is the root cause of the zero sensitivities");
                } else {
                    panic!("Unexpected simulation error: {e:?}");
                }
            }
        }
    }

    /// Creates a test EnzymeML document representing a Michaelis-Menten system
    /// with KM = 100.0 and Vmax = 10.0
    ///
    /// The system models the reaction: S -> P
    /// where the reaction rate follows Michaelis-Menten kinetics:
    /// v = -Vmax * [S] / (KM + [S])
    fn create_enzmldoc() -> Result<EnzymeMLDocument, EnzymeMLDocumentBuilderError> {
        EnzymeMLDocumentBuilder::default()
            .name("test")
            // Define the ODE for substrate concentration
            .to_equations(
                EquationBuilder::default()
                    .species_id("substrate")
                    .equation("-v_max * substrate / (K_M + substrate)") // Michaelis-Menten equation
                    .equation_type(EquationType::Ode)
                    .build()
                    .unwrap(),
            )
            // Define Vmax parameter
            .to_parameters(
                ParameterBuilder::default()
                    .id("v_max")
                    .name("v_max")
                    .symbol("v_max")
                    .value(10.0)
                    .build()
                    .unwrap(),
            )
            // Define KM parameter
            .to_parameters(
                ParameterBuilder::default()
                    .id("K_M")
                    .name("K_M")
                    .symbol("K_M")
                    .value(100.0)
                    .build()
                    .unwrap(),
            )
            .build()
    }

    /// Adds mass conservation assignments to the EnzymeML document.
    ///
    /// Adds two equations:
    /// 1. S_total = substrate + product (initial assignment)
    /// 2. product = S_total - substrate (dynamic assignment)
    ///
    /// This ensures that the total amount of substrate + product remains constant
    /// throughout the simulation.
    fn add_assignments(enzmldoc: &mut EnzymeMLDocument) {
        // Add initial assignment for product
        enzmldoc.equations.push(
            EquationBuilder::default()
                .species_id("S_total")
                .equation("substrate + product")
                .equation_type(EquationType::InitialAssignment)
                .build()
                .unwrap(),
        );

        enzmldoc.equations.push(
            EquationBuilder::default()
                .species_id("product")
                .equation("S_total - substrate")
                .equation_type(EquationType::Assignment)
                .build()
                .unwrap(),
        );
    }

    /// Implementation of a Michaelis-Menten system with analytical sensitivity analysis.
    ///
    /// This serves as a reference implementation to validate the EnzymeML-based system.
    /// It includes both the basic kinetic equations and their derivatives for sensitivity analysis.
    struct MentenSystem {
        v: Box<dyn Fn(f64) -> f64>,     // Rate equation
        dvds: Box<dyn Fn(f64) -> f64>,  // Derivative with respect to substrate
        dvdkm: Box<dyn Fn(f64) -> f64>, // Derivative with respect to KM
        dvmax: Box<dyn Fn(f64) -> f64>, // Derivative with respect to Vmax
    }

    impl MentenSystem {
        fn new(km: f64, v_max: f64) -> Self {
            Self {
                v: Box::new(move |s| -v_max * s / (km + s)),
                dvds: Box::new(move |s| -v_max * km / (km + s).powi(2)),
                dvdkm: Box::new(move |s| v_max * s / (km + s).powi(2)),
                dvmax: Box::new(move |s| -s / (km + s)),
            }
        }

        fn integrate(self, s0: f64, t0: f64, t1: f64, dt: f64) -> StepperOutput {
            let initial_state = vec![s0, 0.0, 0.0];
            let solver = BasicODESolver::new(RK5);
            let (_, y_out) = solver
                .solve(&self, (t0, t1), dt, &initial_state)
                .expect("Integration failed");

            y_out.to_vec()
        }
    }

    impl Default for MentenSystem {
        fn default() -> Self {
            Self::new(100.0, 10.0)
        }
    }

    impl ODEProblem for MentenSystem {
        fn rhs(&self, _t: f64, y: &[f64], dy: &mut [f64]) -> Result<(), argmin_math::Error> {
            let s = y[0];
            let dsdkm = y[1];
            let dsdvmax = y[2];

            // Perform the rhs step
            dy[0] = (self.v)(s);

            // Calculate sensitivities
            // dsdkm
            dy[1] = (self.dvds)(s) * dsdkm + (self.dvdkm)(s);
            // dvmax
            dy[2] = (self.dvds)(s) * dsdvmax + (self.dvmax)(s);

            Ok(())
        }
    }

    // /// Michaelis-Menten turnover of ABTS by SLAC **plus** first-order SLAC inactivation.
    // ///
    // /// State vector layout (8 entries):
    // ///   0  [ABTS]          (a)
    // ///   1  [SLAC]          (s)
    // ///   2  ∂a/∂k_ie
    // ///   3  ∂a/∂K_m
    // ///   4  ∂a/∂k_cat
    // ///   5  ∂s/∂k_ie
    // ///   6  ∂s/∂K_m
    // ///   7  ∂s/∂k_cat
    // pub struct AbtsSlacSystem {
    //     // --- closures for the RHS and its analytical derivatives
    //     v_abts: Box<dyn Fn(f64, f64) -> f64>,   // f₁(a,s)
    //     dv_da: Box<dyn Fn(f64, f64) -> f64>,    // ∂f₁/∂a
    //     dv_ds: Box<dyn Fn(f64, f64) -> f64>,    // ∂f₁/∂s
    //     dv_dkm: Box<dyn Fn(f64, f64) -> f64>,   // ∂f₁/∂K_m
    //     dv_dkcat: Box<dyn Fn(f64, f64) -> f64>, // ∂f₁/∂k_cat
    //     v_slac: Box<dyn Fn(f64) -> f64>,        // f₂(s)
    //     df2_ds: Box<dyn Fn() -> f64>,           // ∂f₂/∂s  (constant −k_ie)
    //     df2_dkie: Box<dyn Fn(f64) -> f64>,      // ∂f₂/∂k_ie  ( = −s )
    // }

    // impl AbtsSlacSystem {
    //     pub fn new(k_ie: f64, k_m: f64, k_cat: f64) -> Self {
    //         // ---- symbols in comments follow the manuscript in the analysis ----------
    //         let v_abts = move |a: f64, s: f64| -k_cat * s * a / (k_m + a);
    //         let dv_da = move |a: f64, s: f64| -k_cat * s * k_m / (k_m + a).powi(2);
    //         let dv_ds = move |a: f64, _s: f64| -k_cat * a / (k_m + a);
    //         let dv_dkm = move |a: f64, s: f64| k_cat * s * a / (k_m + a).powi(2);
    //         let dv_dkcat = move |a: f64, s: f64| -s * a / (k_m + a);

    //         let v_slac = move |s: f64| -k_ie * s;
    //         let df2_ds = move || -k_ie;
    //         let df2_dkie = move |s: f64| -s;

    //         Self {
    //             v_abts: Box::new(v_abts),
    //             dv_da: Box::new(dv_da),
    //             dv_ds: Box::new(dv_ds),
    //             dv_dkm: Box::new(dv_dkm),
    //             dv_dkcat: Box::new(dv_dkcat),
    //             v_slac: Box::new(v_slac),
    //             df2_ds: Box::new(df2_ds),
    //             df2_dkie: Box::new(df2_dkie),
    //         }
    //     }

    //     /// Convenience wrapper – integrates states **and** sensitivities.
    //     pub fn integrate(&self, a0: f64, s0: f64, t0: f64, t1: f64, dt: f64) -> StepperOutput {
    //         let y0 = vec![a0, s0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    //         let solver = BasicODESolver::new(RK5);
    //         let (_, y_out) = solver
    //             .solve(self, (t0, t1), dt, &y0)
    //             .expect("integration failed");
    //         y_out.to_vec()
    //     }
    // }

    // impl ODEProblem for AbtsSlacSystem {
    //     fn rhs(&self, _t: f64, y: &[f64], dy: &mut [f64]) -> Result<(), argmin_math::Error> {
    //         // --- unpack states ------------------------------------------------------
    //         let a = y[0];
    //         let s = y[1];
    //         let da_dkie = y[2];
    //         let da_dkm = y[3];
    //         let da_dkcat = y[4];
    //         let ds_dkie = y[5];
    //         let ds_dkm = y[6];
    //         let ds_dkcat = y[7];

    //         // --- original ODEs ------------------------------------------------------
    //         dy[0] = (self.v_abts)(a, s); // d[ABTS]/dt
    //         dy[1] = (self.v_slac)(s); // d[SLAC]/dt

    //         // --- common partials used several times --------------------------------
    //         let df1_da = (self.dv_da)(a, s);
    //         let df1_ds = (self.dv_ds)(a, s);
    //         let df2_ds = (self.df2_ds)();

    //         // --- sensitivities: chain rule -----------------------------------------
    //         // w.r.t. k_ie ------------------------------------------------------------
    //         dy[2] = df1_da * da_dkie + df1_ds * ds_dkie /* + ∂f₁/∂k_ie (=0) */;
    //         dy[5] = df2_ds * ds_dkie + (self.df2_dkie)(s);

    //         // w.r.t. K_m -------------------------------------------------------------
    //         dy[3] = df1_da * da_dkm + df1_ds * ds_dkm + (self.dv_dkm)(a, s);
    //         dy[6] = df2_ds * ds_dkm; // ∂f₂/∂K_m = 0

    //         // w.r.t. k_cat -----------------------------------------------------------
    //         dy[4] = df1_da * da_dkcat + df1_ds * ds_dkcat + (self.dv_dkcat)(a, s);
    //         dy[7] = df2_ds * ds_dkcat; // ∂f₂/∂k_cat = 0

    //         Ok(())
    //     }
    // }
}
