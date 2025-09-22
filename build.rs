//! Build script to generate Rust code from EnzymeML specifications
//!
//! This script reads markdown specification files from specs/specifications/
//! and generates corresponding Rust code in src/versions/.
//! The code is only regenerated if the specification files have changed.

use mdmodels::prelude::{DataModel, Templates};
use std::{fs, path::PathBuf};

fn main() {
    // Tell cargo to rerun this script only if something in specs/specifications changes
    println!("cargo:rerun-if-changed=specs/specifications");

    // Parse the specs and generate Rust code
    let pattern = "specs/specifications/*.md";
    let files = glob::glob(pattern).expect("Failed to read glob pattern");

    for path in files.flatten() {
        let fname = path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .split('.')
            .next()
            .unwrap();

        // Parse the markdown file into a DataModel
        let mut model = DataModel::from_markdown(&path).expect("Failed to parse markdown");

        // Generate compliant Rust code
        let code = model
            .convert_to(&Templates::Rust, None)
            .expect("Failed to convert to Rust")
            + "\n";

        // Create a new file with the same name as the markdown file
        let out_path: PathBuf = format!("src/versions/{fname}.rs").into();

        // If the file doesn't exist, write the code to the file
        if !out_path.exists() {
            fs::write(&out_path, code).expect("Failed to write file");
            println!("cargo:warning=Updated file {}", out_path.display());
        }
    }
}
