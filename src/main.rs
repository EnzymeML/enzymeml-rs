use std::{io::Write, path::Path};

use enzymeml::{
    optim::{
        observer::CallbackObserver,
        problem::{LBFGMoreThuenteArgs, ProblemBuilder, Solver, Transformation},
        runner::optimize,
    },
    prelude::*,
};
use setup::SimulationSetupBuilder;

pub fn main() {
    let path = Path::new("abts.json");
    let file = std::fs::File::open(path).unwrap();
    let doc: EnzymeMLDocument = serde_json::from_reader(file).unwrap();

    let problem = ProblemBuilder::default()
        .doc(doc.clone())
        .simulation_setup(
            SimulationSetupBuilder::default()
                .dt(1.0)
                .rtol(1e-6)
                .atol(1e-6)
                .build()
                .unwrap(),
        )
        .solver(Solver::LBFGSMoreThuente(LBFGMoreThuenteArgs::default()))
        .transform(Transformation::Pow("k_cat".to_string(), 2.0))
        .transform(Transformation::Pow("k_ie".to_string(), 2.0))
        .build()
        .unwrap();

    let report = optimize(problem, Some(create_file_observer()), true);

    if let Ok(report) = report {
        println!("{:#?}", report.parameter_correlations);
    }
}

fn create_file_observer() -> CallbackObserver {
    let path = "progress.txt";
    CallbackObserver {
        callback: Box::new(move |cost, best_cost| {
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .unwrap();
            writeln!(file, "cost: {:#?} best_cost: {:#?}", cost, best_cost).unwrap();
        }),
    }
}
