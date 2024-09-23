use std::path::Path;

use enzymeml::prelude::*;

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new("model.md");
    let enzmldoc: EnzymeMLDocument = serde_json::from_reader(std::fs::File::open(path)?)?;

    println!("{:#?}", enzmldoc);

    Ok(())
}
