//! Interpreter that emits a pretty-printed version of the program source code.
//! This is a warm-up exercise for the actual interpreter.

use std::fmt::{Debug, Formatter, Pointer};

use crate::il::{IdentifierList, PascalExpr};

use super::super::il::ProgramExpr;

/// Execute the pretty-printing interpreter.
///
/// This interprets the program to build a "print-language"
/// representation of the program.
/// Then it reduces this representation to a string.
pub(crate) fn pretty_print(program: ProgramExpr, indent: usize) -> String {
    let expr = PascalExpr::Program(program);
    let pl: PrintProgram<PrettyPrintContext> = print_program_from_pascal(&expr);
    let start_state = PrettyPrintContext::new(indent);
    let end_state: PrettyPrintContext = interpret_print_program(&pl, start_state);
    end_state.output
}

/// Interpret the print-language program.
///
/// Compare to the Haskell signature: `PrintProgram<'a> -> 'a`
fn interpret_print_program(
    pp: &PrintProgram<PrettyPrintContext>,
    ctx: PrettyPrintContext,
) -> PrettyPrintContext {
    match pp {
        /// The `Pure` constructor in Free Monad terminology.
        // TODO: we should probably use the inside ctx not the function-arg ctx?
        PrintProgram::Stop(_) => ctx,
        /// The `Free` constructor in Free Monad terminology
        PrintProgram::KeepGoing(pi) => match pi {
            PrintInstruction::Write(s, k) => {
                let next_ctx = ctx.write(s);
                let next_pl = k;
                interpret_print_program(&next_pl, next_ctx)
            }
            PrintInstruction::WriteLn(s, k) => {
                let next_ctx = ctx.write_line(s);
                let next_pl = k;
                interpret_print_program(&next_pl, next_ctx)
            }
            PrintInstruction::IncIndent(_, k) => {
                let next_ctx = ctx.increase_indent();
                let next_pl = k;
                interpret_print_program(&next_pl, next_ctx)
            }
            PrintInstruction::DecIndent(_, k) => {
                let next_ctx = ctx
                    .decrease_indent()
                    .expect("decreasing indentation level should not go below zero");
                let next_pl = k;
                interpret_print_program(&next_pl, next_ctx)
            }
            _ => todo!("eval_print_language not implemented for {:?}", pi),
        },
    }
}

/// A program we can use to build up the pretty-printing output.
///
/// The "instruction" is the Functor in the corresponding Free Monad "program", [PrintProgram] .
///
/// A Functor is a type that can be mapped over, or in other words, a data structure
/// with a `map` operation that preserves structure while applying a function to each
/// element of that structure.
///
/// In Haskell the type signature is: `Functor f => a -> f b`, and the corresponding
/// `map` function is `fmap :: (a -> b) -> (f a -> f b)`.
///
/// We are using the terminology from Scott Wlaschin's, F# for Fun and Profit,
/// https://fsharpforfunandprofit.com/posts/13-ways-of-looking-at-a-turtle-2/#way13
/// to describe the interpreter.
///
/// Mark Seemann has also covered this in his blog post, https://blog.ploeh.dk/2018/06/18/free-monad-in-c/
enum PrintInstruction<TNext> {
    // First arg is the input params, second arg is the response function
    // Note that the continuations (second arg) are degenerate functions of
    // no arguments since the operations are not returning any values.
    // We can write them as `Box<dyn Fn() -> TNext>` or more simply
    // `Box<TNext>`.
    Write(String, PICont<TNext>),
    WriteLn(String, PICont<TNext>),
    IncIndent((), PICont<TNext>),
    DecIndent((), PICont<TNext>),
}

//type PICont<TNext> = Box<dyn Fn() -> TNext>;
type PICont<TNext> = Box<TNext>;

/*
impl<'a, TNext> PrintInstruction<'a, TNext> {
    /// Map the functor over the continuation function
    fn map_instr(&self, f: Box<dyn Fn(TNext) -> TNext>) -> PrintInstruction<TNext> {
        match self {
            PrintInstruction::Write(s, k) => PrintInstruction::Write(s.clone(), Box::new(|| f(k()))),
            PrintInstruction::WriteLn(s, k) => PrintInstruction::WriteLn(s.clone(), Box::new(|| f(k()))),
            PrintInstruction::IncIndent(_, k) => PrintInstruction::IncIndent((), Box::new(|| f(k()))),
            PrintInstruction::DecIndent(_, k) => PrintInstruction::DecIndent((), Box::new(|| f(k()))),
        }
    }
}
*/

impl<TNext> Debug for PrintInstruction<TNext> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self {
            PrintInstruction::Write(s, _) => f.write_fmt(format_args!("Write({:?})", s)),
            PrintInstruction::WriteLn(s, _) => f.write_fmt(format_args!("WriteLn({:?})", s)),
            PrintInstruction::IncIndent(_, _) => f.write_str("IncIndent"),
            PrintInstruction::DecIndent(_, _) => f.write_str("DecIndent"),
        }
    }
}

/// Monadic wrapper type for [PrintInstruction] that allows the program to stop or continue.
///
/// The "program" ([PrintProgram]) is the Free Monad for the [PrintInstruction] Functor.
///
/// Notice how it looks a lot like the [Result] type, *e.g.*
/// [Result<PrintInstruction<PrintProgram<T>>, T>] or even the [Option] type, if we ignore that `Stop`
/// takes a `T`.
///
/// # Free Monad in Haskell
/// In Haskell the type signatures are:
/// `Pure :: a -> Free f a`
/// `Free :: f (Free f a) -> Free f a`

// TODO: consider using the Result type to get a lot of functions for free
enum PrintProgram<T> {
    /// The `Pure` constructor in Free Monad terminology.
    Stop(T),
    /// The `Free` constructor in Free Monad terminology
    KeepGoing(PrintInstruction<PrintProgram<T>>),
}

impl<T> PrintProgram<T>
where
    T: Default,
{
    /// Stop constructor
    fn stop() -> Self {
        PrintProgram::Stop(Default::default())
    }
    /// Keep going and write constructor
    fn write(s: String, k: Self) -> Self {
        PrintProgram::KeepGoing(PrintInstruction::Write(s, Box::new(k)))
    }
    /// Keep going and write_ln constructor
    fn write_ln(s: String, k: Self) -> Self {
        PrintProgram::KeepGoing(PrintInstruction::WriteLn(s, Box::new(k)))
    }

    /// Keep going and increase indentation constructor
    fn inc_indent(k: Self) -> Self {
        PrintProgram::KeepGoing(PrintInstruction::IncIndent((), Box::new(k)))
    }

    /// Keep going and decrease indentation constructor
    fn dec_indent(k: Self) -> Self {
        PrintProgram::KeepGoing(PrintInstruction::DecIndent((), Box::new(k)))
    }
}

/// The context (state) for the pretty-printing interpreter.
#[derive(Debug)]
struct PrettyPrintContext {
    indent_step: usize,
    current_indent_level: usize,
    output: String,
}

impl Default for PrettyPrintContext {
    /// Instantiate a default context with 4 spaces per indent level.
    fn default() -> Self {
        Self::new(4)
    }
}

impl PrettyPrintContext {
    fn new(spaces_per_indent: usize) -> Self {
        Self {
            indent_step: spaces_per_indent,
            current_indent_level: 0,
            output: String::new(),
        }
    }
    /// Return the indent string as spaces
    fn indentation(&self) -> String {
        " ".repeat(self.indent_step)
            .repeat(self.current_indent_level)
    }

    /// Write a string to the output
    fn write(mut self, s: &str) -> Self {
        self.output.push_str(s);
        self
    }

    /// Add a line to the output
    /// The line is indented according to the current indentation level
    fn write_line(mut self, line: &String) -> Self {
        self.output.push_str(line);
        self.output.push_str("\n");
        self.output.push_str(&self.indentation());
        self
    }

    fn increase_indent(mut self) -> Self {
        self.current_indent_level += 1;
        self
    }
    fn decrease_indent(mut self) -> Result<Self, ()> {
        match self.current_indent_level {
            0 => Err(()),
            _ => {
                self.current_indent_level -= 1;
                Ok(self)
            }
        }
    }
}

/// Translate the Pascal expression into a print-language expression.
fn print_program_from_pascal<TNext>(pascal: &PascalExpr) -> PrintProgram<TNext>
where
    TNext: Default,
{
    match pascal {
        PascalExpr::Program(p) => print_program_from_program(p),
        PascalExpr::IdentifierList(il) => todo!(),
        PascalExpr::Declarations(ds) => todo!(),
        PascalExpr::SubprogramDeclarations(sd) => todo!(),
        PascalExpr::CompoundStatement(cs) => todo!(),
    }
}

fn print_program_from_program<TNext>(p: &ProgramExpr) -> PrintProgram<TNext>
where
    TNext: Default,
{
    match p {
        ProgramExpr {
            id,
            identifier_list,
            declarations,
            subprogram_declarations,
            compound_statement,
        } => {
            PrintProgram::write(
                format!("program {}(", id.to_string()),
                PrintProgram::write(
                    format_identifier_list(identifier_list),
                    PrintProgram::write_ln(
                        ");".to_string(),
                        PrintProgram::inc_indent(PrintProgram::write_ln(
                            "begin".to_string(),
                            PrintProgram::write(
                                "{ compound statement }".to_string(),
                                PrintProgram::dec_indent(PrintProgram::write_ln(
                                    "".to_string(),
                                    PrintProgram::write_ln(
                                        "end.".to_string(),
                                        PrintProgram::stop(),
                                    ),
                                )),
                            ),
                        )),
                    ),
                ),
            )
        }
    }
}

fn format_identifier_list(il: &IdentifierList) -> String {
    il.0 .0
        .iter()
        .map(|id| id.to_string())
        .collect::<Vec<String>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use crate::examples::hello_world;

    use super::*;

    #[test]
    fn pretty_print_hello_world() {
        let p = hello_world();
        let actual = pretty_print(p, 2);
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