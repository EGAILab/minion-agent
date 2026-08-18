use std::env;
use std::process::ExitCode;

fn main() -> ExitCode {
    match env::args().nth(1).as_deref() {
        Some("help") | None => {
            println!("Usage: cargo run -p xtask -- <command>");
            println!("Commands: help, conformance, coverage, layering");
            ExitCode::SUCCESS
        }
        Some("conformance" | "coverage" | "layering") => ExitCode::SUCCESS,
        Some(command) => {
            eprintln!("unknown command: {command}");
            ExitCode::FAILURE
        }
    }
}
