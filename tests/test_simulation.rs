#[cfg(not(feature = "wasm"))]
#[cfg(test)]
mod test_simulation {
    use std::collections::HashMap;

    use approx::assert_relative_eq;
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
            initial_conditions.clone(),
            None,
            None,
            RK5::default(),
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
                RK5::default(),
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
            initial_conditions.clone(),
            None,
            None,
            RK5::default(),
            Some(Mode::Sensitivity),
        );

        // ASSERT
        // Compare sensitivity analysis results between both implementations
        if let Ok(MatrixResult {
            species: _,
            parameter_sensitivities,
            times: _,
            assignments: _,
        }) = result
        {
            if let Some(parameter_sensitivities) = parameter_sensitivities {
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
            initial_conditions.clone(),
            None,
            None,
            RK5::default(),
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
                initial_conditions.clone(),
                None,
                None,
                RK5::default(),
                Some(Mode::Regular),
            )
            .expect("Simulation failed");

        let plot = result.plot(PlotConfig::default(), false);
        plot.to_html();
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
                    .equation_type(EquationType::ODE)
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
                .equation_type(EquationType::INITIAL_ASSIGNMENT)
                .build()
                .unwrap(),
        );

        enzmldoc.equations.push(
            EquationBuilder::default()
                .species_id("product")
                .equation("S_total - substrate")
                .equation_type(EquationType::ASSIGNMENT)
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
            let solver = BasicODESolver::new(RK5::default());
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
}
