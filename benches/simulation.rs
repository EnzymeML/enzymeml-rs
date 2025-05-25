use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use enzymeml::prelude::*;

fn setup_simulation(reaction: bool) -> (ODESystem, Vec<InitialCondition>, Vec<SimulationSetup>) {
    // First, we create the EnzymeMLDocument that contains the model and the measurements
    let doc = if reaction {
        get_reaction_doc()
    } else {
        get_doc()
    };

    let measurement = doc.measurements.first().unwrap();
    let initial_condition: InitialCondition = measurement.into();
    let initial_conditions = vec![initial_condition; 120];

    let system: ODESystem = doc.try_into().unwrap();

    let setup = SimulationSetupBuilder::default()
        .dt(1.0)
        .t0(0.0)
        .t1(100.0)
        .build()
        .expect("Failed to build simulation setup");

    let setups = vec![setup; 120];

    (system, initial_conditions, setups)
}

fn benchmark_simulation(c: &mut Criterion) {
    let (system, initial_conditions, setups) = setup_simulation(false);
    let (system_reaction, initial_conditions_reaction, setups_reaction) = setup_simulation(true);

    c.bench_function("ode_integration", |b| {
        b.iter(|| {
            let _ = black_box(system.integrate::<SimulationResult>(
                black_box(&setups[0]),
                black_box(&initial_conditions[0]),
                None,
                None,
                RK4,
                Some(Mode::Regular),
            ));
        });
    });

    c.bench_function("reaction_integration", |b| {
        b.iter(|| {
            let _ = black_box(system_reaction.integrate::<SimulationResult>(
                black_box(&setups_reaction[0]),
                black_box(&initial_conditions_reaction[0]),
                None,
                None,
                RK4,
                Some(Mode::Regular),
            ));
        });
    });

    c.bench_function("bulk_ode_integration", |b| {
        b.iter(|| {
            let _ = black_box(system.bulk_integrate::<SimulationResult>(
                black_box(&setups),
                black_box(&initial_conditions),
                None,
                None,
                RK4,
                Some(Mode::Regular),
            ));
        });
    });

    c.bench_function("bulk_reaction_integration", |b| {
        b.iter(|| {
            let _ = black_box(system_reaction.bulk_integrate::<SimulationResult>(
                black_box(&setups_reaction),
                black_box(&initial_conditions_reaction),
                None,
                None,
                RK4,
                Some(Mode::Regular),
            ));
        });
    });
}

criterion_group!(benches, benchmark_simulation);
criterion_main!(benches);

fn get_doc() -> EnzymeMLDocument {
    EnzymeMLDocumentBuilder::default()
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
        .build()
        .expect("Failed to build document")
}

fn get_reaction_doc() -> EnzymeMLDocument {
    EnzymeMLDocumentBuilder::default()
        .name("Michaelis-Menten Simulation")
        .to_reactions(
            ReactionBuilder::default()
                .name("Michaelis-Menten")
                .id("michaelis-menten")
                .reversible(false)
                .to_reactants(
                    ReactionElementBuilder::default()
                        .species_id("substrate")
                        .stoichiometry(1)
                        .build()
                        .expect("Failed to build reaction element"),
                )
                .to_products(
                    ReactionElementBuilder::default()
                        .species_id("product")
                        .stoichiometry(1)
                        .build()
                        .expect("Failed to build reaction element"),
                )
                .kinetic_law(
                    EquationBuilder::default()
                        .species_id("substrate")
                        .equation("-v_max * substrate / (K_M + substrate)")
                        .equation_type(EquationType::RateLaw)
                        .build()
                        .expect("Failed to build kinetic law"),
                )
                .build()
                .expect("Failed to build reaction"),
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
        .build()
        .expect("Failed to build document")
}
