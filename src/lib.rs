//! Main entry point for the underlying library.

use std::error::Error;
mod examples;
mod front_end;
mod il;
mod interpreters;

pub fn pretty_print_command(fname: &String) -> Result<(), Box<dyn Error>> {
    println!("{}", pretty_print_file(fname)?);
    Ok(())
}

/// Load an pretty-print a file at the given path.
pub fn pretty_print_file(fname: &String) -> Result<String, Box<dyn Error>> {
    let content = std::fs::read_to_string(fname)?;
    let program = front_end::parse_program_string(&content)?;
    let pretty_printed = interpreters::pretty_printer::pretty_print(program, 2);
    Ok(pretty_printed)
}
