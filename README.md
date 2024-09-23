# EnzymeML-Rust

This is the official [EnzymeML](https://enzymeml.org) Rust library.

> [!WARNING]
> The library is currently under development and is not yet ready for production use.

## Installation

```bash
cargo install enzymeml
```

## Examples

### Create an EnzymeML document

````rust
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
````

### Deserialize an EnzymeML document

```rust
use enzymeml::prelude::*;
use std::path::Path;

let path = Path::new("model.md");
let enzmldoc: EnzymeMLDocument = serde_json::from_reader(std::fs::File::open(path)?)?;

println!("{:#?}", enzmldoc);
```
