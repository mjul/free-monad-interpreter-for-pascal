//! Interpreter that emits a pretty-printed version of the program source code.
//! This is a warm-up exercise for the actual interpreter.

use super::super::il::ProgramExpr;

/// Execute the pretty-printing interpreter.
pub(crate) fn pretty_print(program: &ProgramExpr, indent: usize) -> String {
    let mut ctx = PrettyPrintContext::new(indent);
    // TODO: call the interpreter
    ctx.output
}


/// The context for the pretty-printing interpreter.
struct PrettyPrintContext {
    indent: usize,
    output: String,
}

impl PrettyPrintContext {
    fn new(indent: usize) -> Self {
        Self {
            indent,
            output: String::new(),
        }
    }
    /// Return the indent string as spaces
    fn indentation(&self) -> String {
        " ".repeat(self.indent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examples::hello_world;
    #[test]
    fn pretty_print_hello_world() {
        let p = hello_world();
        let actual = pretty_print(&p, 4);
        let expected = r#"
program helloWorld(output);
begin
    writeLn('Hello, World!')
end."#
            .to_string()
            .trim()
            .to_string();

        assert_eq!(expected, actual);
    }
}