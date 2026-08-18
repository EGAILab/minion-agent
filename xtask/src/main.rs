use std::env;
use std::ffi::OsStr;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<_> = env::args_os().skip(1).collect();
    match args.first().map(|argument| argument.as_os_str()) {
        Some(command) if command == OsStr::new("help") => {
            println!("Usage: cargo run -p xtask -- <command>");
            println!("Commands: help, conformance, coverage, layering");
            ExitCode::SUCCESS
        }
        Some(command) if command == OsStr::new("conformance") => {
            match xtask::conformance::run_cli(&args[1..]) {
                Ok(()) => ExitCode::SUCCESS,
                Err(error) => {
                    eprintln!("{error:#}");
                    ExitCode::FAILURE
                }
            }
        }
        Some(command) if command == OsStr::new("coverage") || command == OsStr::new("layering") => {
            ExitCode::SUCCESS
        }
        None => {
            eprintln!("missing command");
            ExitCode::FAILURE
        }
        Some(command) => {
            eprintln!("unknown command: {}", command.to_string_lossy());
            ExitCode::FAILURE
        }
    }
}
