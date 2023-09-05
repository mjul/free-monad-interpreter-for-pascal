use std::env;
use std::error::Error;

mod front_end;
mod il;
mod interpreters;
mod examples;


fn pretty_print_command(fname:&String) -> Result<(), Box<dyn Error>> {
    let content = std::fs::read_to_string(fname)?;
    let program = front_end::parse_program_string(&content)?;
    let pretty_printed = interpreters::pretty_printer::pretty_print(program, 2);
    println!("{}", pretty_printed);
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().collect();
    println!("-->> {:?}", args);
    return match &args[..] {
        [_, fname] => pretty_print_command(fname),
        _ => {
            println!("Usage: {} <filename>", args[0]);
            Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, "Invalid arguments")))
        }
    };
}
