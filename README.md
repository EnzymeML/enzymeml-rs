# 🧪 EnzymeML-Rust

<div align="center">

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status: In Development](https://img.shields.io/badge/Status-In%20Development-blue)
[![Crates.io](https://img.shields.io/crates/v/enzymeml.svg)](https://crates.io/crates/enzymeml)

**The official [EnzymeML](https://enzymeml.org) toolkit for Rust** - Powerful enzyme kinetics modeling and simulation

</div>

> ⚠️ **This library is currently under development and is not yet ready for production use.**

## ✨ Features

- 📄 **EnzymeML Document Management** - Create, parse, and manipulate EnzymeML documents
- 🧮 **Simulation** - Simulate enzyme kinetics through ODE systems with various solvers
- 📊 **Optimization** - Parameter estimation and model fitting with multiple algorithms
- ✅ **Validation** - Ensure models are consistent and correct
- 📋 **Data Handling** - Read/write measurement data in various tabular formats
- 📈 **Visualization** - Beautiful plots for simulation results and experimental data
- 📊 **SBML Support** - Read and write SBML documents
- 🌐 **WebAssembly Support** - Use in web applications

## 🚀 Installation

```bash
cargo add enzymeml
```

Or add to your `Cargo.toml`:

```toml
[dependencies]
enzymeml = "0.1.0"
```

## 🔍 Usage Examples

### 🧪 Creating an EnzymeML Document

```rust
use enzymeml::prelude::*;

let mut enzmldoc = EnzymeMLDocumentBuilder::default();

// Create small molecules
let substrate = SmallMoleculeBuilder::default()
    .id("s1")
    .name("Substrate")
    .build()?;
let product = SmallMoleculeBuilder::default()
    .id("s2")
    .name("Product")
    .build()?;

enzmldoc.to_small_molecules(substrate);
enzmldoc.to_small_molecules(product);

// Create a reaction
let reaction = build_reaction!(
    "r1",
    "Reaction",
    true,
    "s1" => -1.0,
    "s2" => 1.0
);

enzmldoc.to_reactions(reaction);

// Create an equation
let equation = EquationBuilder::default()
    .species_id("s1")
    .equation("v_max * s1 / (k_m + s1)")
    .build()?;

enzmldoc.to_equations(equation);

// Serialize the document
let enzmldoc = enzmldoc.build()?;
let serialized = serde_json::to_string_pretty(&enzmldoc)?;

println!("{}", serialized);
```

### 📥 Deserializing an EnzymeML Document

```rust
use enzymeml::prelude::*;
use std::path::Path;

let path = Path::new("model.json");
let enzmldoc: EnzymeMLDocument = read_enzmldoc(path).unwrap();

println!("{:#?}", enzmldoc);
```

### 🧮 Simulating Enzyme Kinetics

```rust
use enzymeml::prelude::*;
use plotly::{Layout, Plot};

// Create an EnzymeML document with Michaelis-Menten kinetics
let doc = EnzymeMLDocumentBuilder::default()
    .name("Michaelis-Menten Simulation")
    .to_equations(
        EquationBuilder::default()
            .species_id("substrate")
            .equation("-v_max * substrate / (K_M + substrate)")
            .equation_type(EquationType::Ode)
            .build()?,
    )
    .to_equations(
        EquationBuilder::default()
            .species_id("product")
            .equation("v_max * substrate / (K_M + substrate)")
            .equation_type(EquationType::Ode)
            .build()?,
    )
    .to_parameters(
        ParameterBuilder::default()
            .name("v_max")
            .id("v_max")
            .symbol("v_max")
            .value(2.0)
            .build()?,
    )
    .to_parameters(
        ParameterBuilder::default()
            .name("K_M")
            .id("K_M")
            .symbol("K_M")
            .value(100.0)
            .build()?,
    )
    .to_measurements(/* Define initial concentrations */)
    .build()?;

// Extract initial conditions from measurements
let measurement = doc.measurements.first().unwrap();
let initial_conditions: InitialCondition = measurement.into();

// Create an ODE system from the EnzymeML document
let system: ODESystem = doc.try_into().unwrap();

// Configure simulation parameters
let setup = SimulationSetupBuilder::default()
    .dt(0.1)
    .t0(0.0)
    .t1(200.0)
    .build()?;

// Run the simulation
let result = system.integrate::<SimulationResult>(
    &setup,
    &initial_conditions,
    None,  // Optional new parameters
    None,  // Optional specific time points
    RK5,   // Numerical solver
    Some(Mode::Regular),
);

// Visualize the results
if let Ok(result) = result {
    let mut plot: Plot = result.into();
    plot.set_layout(
        Layout::default()
            .title("Michaelis-Menten Simulation")
            .show_legend(true),
    );
    plot.show();
}
```

## 🧩 Available Features

Enable specific features in your `Cargo.toml`:

```toml
enzymeml = { version = "0.1.0", features = ["optimization", "simulation"] }
```

| Feature        | Description                              |
| -------------- | ---------------------------------------- |
| `simulation`   | 🧮 ODE simulation capabilities            |
| `optimization` | 📊 Parameter estimation and model fitting |
| `tabular`      | 📋 Reading/writing tabular data           |
| `llm`          | 🤖 Large language model integration       |
| `wasm`         | 🌐 WebAssembly support                    |

## 📚 Documentation

For complete documentation, check out the [API docs](https://docs.rs/enzymeml).

## 📜 License

This project is licensed under the MIT License.

---

<div align="center">
<strong>Made with ❤️ by the EnzymeML Team</strong>
</div>