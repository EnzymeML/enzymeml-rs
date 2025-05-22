use enzymeml::prelude::*;
use plotly::{Layout, Plot};

fn main() -> Result<(), EnzymeMLDocumentBuilderError> {
    // First, we create the EnzymeMLDocument that contains the model and the measurements
    // We leave the measurement data empty for now, because we are only interested in deriving
    // the initial conditions from the measurements
    let doc = EnzymeMLDocumentBuilder::default()
        .name("Michaelis-Menten Simulation")
        .to_equations(
            EquationBuilder::default()
                .species_id("substrate")
                .equation("-v_max * substrate / (K_M + substrate)")
                .equation_type(EquationType::Ode)
                .build()
                .expect("Failed to build equation"),
        )
        .to_equations(
            EquationBuilder::default()
                .species_id("product")
                .equation("v_max * substrate / (K_M + substrate)")
                .equation_type(EquationType::Ode)
                .build()
                .expect("Failed to build equation"),
        )
        .to_parameters(
            ParameterBuilder::default()
                .name("v_max")
                .id("v_max")
                .symbol("v_max")
                .value(2.0)
                .build()
                .expect("Failed to build parameter"),
        )
        .to_parameters(
            ParameterBuilder::default()
                .name("K_M")
                .id("K_M")
                .symbol("K_M")
                .value(100.0)
                .build()
                .expect("Failed to build parameter"),
        )
        .to_measurements(
            MeasurementBuilder::default()
                .name("measurement")
                .id("measurement")
                .to_species_data(
                    MeasurementDataBuilder::default()
                        .species_id("substrate")
                        .initial(100.0)
                        .build()
                        .expect("Failed to build measurement data"),
                )
                .to_species_data(
                    MeasurementDataBuilder::default()
                        .species_id("product")
                        .initial(0.0)
                        .build()
                        .expect("Failed to build measurement data"),
                )
                .build()
                .expect("Failed to build measurement"),
        )
        .build()?;

    // In order to retrieve the initial conditions from the measurements, we first need to
    // convert the Measurement into an InitialCondition type
    //
    // This will automatically extract the species IDs and their corresponding initial values
    // from the Measurement and return them as a HashMap (aliased by "InitialCondition")
    let measurement = doc.measurements.first().unwrap();
    let initial_conditions: InitialCondition = measurement.into();

    // Next, we utilize Rust's type inference to convert the EnzymeMLDocument into an ODESystem
    // The ODE-System is the core of the simulation and contains the following information:
    //
    // - JIT-compiled model equations
    // - JIT-compiled derivatives (wrt to y and wrt to θ)
    //
    // This allows you to use the ODE-System for simulation, optimization,
    // and sensitivity analysis
    let system: ODESystem = doc.try_into().unwrap();

    // Before we can simulate the model, we need to create a SimulationSetup that contains
    // the time parameters for the simulation.
    //
    // Alternatively, we can also use Rusts type inference
    // to convert a single Measurement into a SimulationSetup
    // This is useful when we want to reproduce a measurement
    let setup = SimulationSetupBuilder::default()
        .dt(0.1)
        .t0(0.0)
        .t1(200.0)
        .build()
        .expect("Failed to build simulation setup");

    // Finally, we can run the simulation!
    //
    // Please note, that you can dictate the output format of the simulation
    // by providing a different struct to the integrate function. We provide
    // MatrixOutput and SimulationResultOutput as default output formats.
    //
    // By implementing the OutputFormat trait, you can create your own custom output formats
    let result = system.integrate::<SimulationResult>(
        &setup,
        initial_conditions,
        None,                // We could also dynamically set new parameters
        None,                // We could also provide specific time points to extract
        RK5,                 // We could also use a different solver
        Some(Mode::Regular), // We could also use a different mode (e.g. Sensitivity)
    );

    if let Ok(result) = result {
        // Again, we can use Rusts type inference to convert the SimulationResult
        // into a Plotly Plot and display it
        let mut plot: Plot = result.into();
        plot.set_layout(
            Layout::default()
                .title("Michaelis-Menten Simulation")
                .show_legend(true),
        );
        plot.show();
    } else {
        println!("Simulation failed");
    }

    Ok(())
}
