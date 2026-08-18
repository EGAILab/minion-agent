use std::env;
use std::process::ExitCode;

fn main() -> ExitCode {
    match env::args().nth(1).as_deref() {
        Some("help") => {
            println!("Usage: cargo run -p xtask -- <command>");
            println!("Commands: help, conformance, coverage, layering");
            ExitCode::SUCCESS
        }
        Some("conformance" | "coverage" | "layering") => ExitCode::SUCCESS,
        None => {
            eprintln!("missing command");
            ExitCode::FAILURE
        }
        Some(command) => {
            eprintln!("unknown command: {command}");
            ExitCode::FAILURE
        }
    }
}
