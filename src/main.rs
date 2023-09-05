//! Main is just a thin shell around the application in defined in `lib.rs`.
//!
//! This is a common pattern in Rust since using `lib.rs` allows
//! the integration tests in `tests\integration_test.rs` to access the code from
//! here. They cannot access the code in `main.rs`.

use std::env;
use std::error::Error;

// Get the commands from the main library entry to the application.

use free_monad_interpreter_for_pascal::pretty_print_command;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().collect();
    println!("-->> {:?}", args);
    return match &args[..] {
        [_, fname] => pretty_print_command(fname),
        _ => {
            println!("Usage: {} <filename>", args[0]);
            Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Invalid arguments",
            )))
        }
    };
}
