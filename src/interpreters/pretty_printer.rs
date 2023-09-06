//! Interpreter that emits a pretty-printed version of the program source code.
//! This is a warm-up exercise for the actual interpreter.
//!
//! The main entry point is the [pretty_print] function which formats a Pascal program AST to a string.
//!
//! It consists of two parts: a recursive function [print_program_from_pascal] that translates the
//! AST to a simple printing language, a Free Monad, and
//! a Free Monad interpreter, [interpret_print_program] that interprets the printing language
//! to produce a string to accomplish the formatting.
//!
//! The translation from AST to printing language is done by a recursive function that is a good
//! example of the continuation passing style (CPS) that is also used in the Free Monad interpreter.
//! Interestingly, CPS allows us to build data structures such as lists and trees top down even
//! though the data structures must be constructed bottom up.

use std::fmt::{Debug, Formatter, Pointer};
use std::ops::Deref;

use crate::il::{
    AssignmentStatement, CompoundStatement, DeclarationsExpr, Expression, ExpressionList, Factor,
    Id, IdentifierList, IfThenElseStatement, MulOp, NonEmptyVec, PascalExpr, ProcedureStatement,
    RelOp, SimpleExpression, StandardType, Statement, SubprogramDeclaration,
    SubprogramDeclarations, Term, Type, VarDeclaration, Variable, WhileDoStatement,
};

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
/// <https://fsharpforfunandprofit.com/posts/13-ways-of-looking-at-a-turtle-2/#way13>
/// to describe the interpreter.
///
/// Mark Seemann has also covered this in his blog post, <https://blog.ploeh.dk/2018/06/18/free-monad-in-c/>
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

// A continuation is normally a function from the current state to the next state.
// However, following the style of the Wlaschin article, we are using a degenerate
// function of no arguments, simplified to a constant. This makes the code simpler
// with relation to the borrow checker since moves into closures are tricky.
//
// It could also be a more elaborate constant function, `type PICont<TNext> = Box<dyn Fn() -> TNext>;`
// or in the general case, `type PICont<TNext> = Box<dyn Fn(TNext) -> TNext>;`
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
        } => PrintProgram::write(
            format!("program {}(", id.to_string()),
            print_program_from_identifier_list(
                identifier_list,
                PrintProgram::write_ln(
                    ");".to_string(),
                    print_program_from_declarations(
                        declarations,
                        print_program_from_subprogram_declarations(
                            subprogram_declarations,
                            print_program_from_compound_statement(
                                compound_statement,
                                PrintProgram::write(".".to_string(), PrintProgram::stop()),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    }
}

fn print_program_from_declarations<TNext>(
    decl_expr: &DeclarationsExpr,
    k: PrintProgram<TNext>,
) -> PrintProgram<TNext>
where
    TNext: Default,
{
    let DeclarationsExpr(vds) = decl_expr;
    match vds.is_empty() {
        // In the empty case we just return the continuation so we don't print anything
        true => k,
        false => print_program_interpose(
            vds,
            &print_program_from_variable_declaration,
            &|k_pi| PrintProgram::write_ln("".to_string(), k_pi),
            k,
        ),
    }
}

fn print_program_from_variable_declaration<TNext>(
    vd: &VarDeclaration,
    k: PrintProgram<TNext>,
) -> PrintProgram<TNext>
where
    TNext: Default,
{
    let VarDeclaration(il, ty) = vd;
    PrintProgram::inc_indent(PrintProgram::write_ln(
        "var".to_string(),
        print_program_from_identifier_list(
            il,
            PrintProgram::write(
                " : ".to_string(),
                print_program_from_type(
                    ty,
                    PrintProgram::dec_indent(PrintProgram::write(";".to_string(), k)),
                ),
            ),
        ),
    ))
}

fn print_program_from_type<TNext>(ty: &Type, k: PrintProgram<TNext>) -> PrintProgram<TNext>
where
    TNext: Default,
{
    match ty {
        Type::StandardType(st) => print_program_from_standard_type(st, k),
    }
}

fn print_program_from_standard_type<TNext>(
    st: &StandardType,
    k: PrintProgram<TNext>,
) -> PrintProgram<TNext>
where
    TNext: Default,
{
    PrintProgram::write(
        match st {
            StandardType::Integer => "integer",
            StandardType::Real => "real",
        }
        .to_string(),
        k,
    )
}

fn print_program_from_subprogram_declarations<TNext>(
    spds: &SubprogramDeclarations,
    k: PrintProgram<TNext>,
) -> PrintProgram<TNext>
where
    TNext: Default,
{
    let SubprogramDeclarations(ds) = spds;
    match ds.is_empty() {
        // In the empty case, there is no semicolon terminator so we just return the continuation
        true => k,
        // In the other cases, add semicolon terminator after each declaration
        false => print_program_terminators(
            ds,
            &print_program_from_subprogram_declaration,
            &|k| PrintProgram::write_ln(";".to_string(), k),
            k,
        ),
    }
}

fn print_program_from_subprogram_declaration<TNext>(
    sd: &SubprogramDeclaration,
    k: PrintProgram<TNext>,
) -> PrintProgram<TNext>
where
    TNext: Default,
{
    todo!("print_program_from_subprogram_declaration");
    k
}

fn print_program_from_compound_statement<TNext>(
    cs: &CompoundStatement,
    k: PrintProgram<TNext>,
) -> PrintProgram<TNext>
where
    TNext: Default,
{
    let CompoundStatement(stmts) = cs;

    PrintProgram::write(
        "begin".to_string(),
        PrintProgram::inc_indent(PrintProgram::write_ln(
            "".to_string(),
            print_program_from_optional_statements(
                stmts,
                PrintProgram::dec_indent(PrintProgram::write_ln(
                    "".to_string(),
                    PrintProgram::write("end".to_string(), k),
                )),
            ),
        )),
    )
}

fn print_program_surround<TNext>(
    header_k: PrintProgram<TNext>,
    body_k: PrintProgram<TNext>,
    tail_k: PrintProgram<TNext>,
) -> PrintProgram<TNext>
where
    TNext: Default,
{
    PrintProgram::stop()
}

/// Build a print program from a slice of `T`s
/// and functions to build the elements of the slice and
/// the interposed elements.
/// For example, build a comma-separated list from a slice of expressions.
fn print_program_interpose<T, TNext>(
    xs: &[T],
    print_x: &dyn Fn(&T, PrintProgram<TNext>) -> PrintProgram<TNext>,
    print_interpose: &dyn Fn(PrintProgram<TNext>) -> PrintProgram<TNext>,
    k: PrintProgram<TNext>,
) -> PrintProgram<TNext>
where
    TNext: Default,
{
    match xs.len() {
        0 => k,
        1 => print_x(&xs[0], k),
        _ => {
            let (head, tail) = xs.split_first().unwrap();
            print_x(
                head,
                print_interpose(print_program_interpose(tail, print_x, print_interpose, k)),
            )
        }
    }
}

/// Interweave a print program to add terminators after every element of a slice,
/// for example to terminate a list of declarations with semicolon.
fn print_program_terminators<T, TNext>(
    xs: &[T],
    print_x: &dyn Fn(&T, PrintProgram<TNext>) -> PrintProgram<TNext>,
    print_interpose: &dyn Fn(PrintProgram<TNext>) -> PrintProgram<TNext>,
    k: PrintProgram<TNext>,
) -> PrintProgram<TNext>
where
    TNext: Default,
{
    // Interpose and add the trailing terminator
    print_program_interpose(xs, print_x, print_interpose, print_interpose(k))
}

fn print_program_from_optional_statements<TNext>(
    stmts: &[Statement],
    k: PrintProgram<TNext>,
) -> PrintProgram<TNext>
where
    TNext: Default,
{
    print_program_interpose(
        stmts,
        &print_program_from_statement,
        &|k| PrintProgram::write_ln(";".to_string(), k),
        k,
    )
}

fn print_program_from_statement<TNext>(
    stmt: &Statement,
    k: PrintProgram<TNext>,
) -> PrintProgram<TNext>
where
    TNext: Default,
{
    match stmt {
        Statement::Assignment(asn) => print_program_from_assignment_statement(asn, k),
        Statement::Procedure(ps) => print_program_from_procedure_statement(ps, k),
        // TODO: print_program_from_compound_statement
        Statement::Compound(cs) => PrintProgram::write("{ compound }".to_string(), k),
        Statement::IfThenElse(ites) => print_program_from_if_then_else_statement(ites, k),
        Statement::WhileDo(wds) => print_program_from_while_do_statement(wds, k),
    }
}

fn print_program_from_assignment_statement<TNext>(
    asn: &AssignmentStatement,
    k: PrintProgram<TNext>,
) -> PrintProgram<TNext>
where
    TNext: Default,
{
    match asn {
        AssignmentStatement(lvar, val) => print_program_from_variable(
            lvar,
            PrintProgram::write(" := ".to_string(), print_program_from_expression(val, k)),
        ),
    }
}

fn print_program_from_variable<TNext>(var: &Variable, k: PrintProgram<TNext>) -> PrintProgram<TNext>
where
    TNext: Default,
{
    match var {
        Variable::Id(id) => print_program_from_id(id, k),
        Variable::ArrayIndex(id, expr) => print_program_from_id(
            id,
            PrintProgram::write(
                "[".to_string(),
                print_program_from_expression(expr, PrintProgram::write("]".to_string(), k)),
            ),
        ),
    }
}

fn print_program_from_procedure_statement<TNext>(
    ps: &ProcedureStatement,
    k: PrintProgram<TNext>,
) -> PrintProgram<TNext>
where
    TNext: Default,
{
    match ps {
        ProcedureStatement(id, None) => PrintProgram::write(id.to_string(), k),
        ProcedureStatement(id, Some(el)) => PrintProgram::write(
            format!("{}(", id.to_string()),
            print_program_from_expression_list(el, PrintProgram::write(")".to_string(), k)),
        ),
    }
}

fn print_program_from_if_then_else_statement<TNext>(
    ites: &IfThenElseStatement,
    k: PrintProgram<TNext>,
) -> PrintProgram<TNext>
where
    TNext: Default,
{
    match ites {
        IfThenElseStatement(cond_expr, then_stmt, else_stmt) => PrintProgram::write(
            "if ".to_string(),
            print_program_from_expression(
                cond_expr,
                PrintProgram::inc_indent(PrintProgram::write_ln(
                    " then".to_string(),
                    print_program_from_statement(
                        then_stmt,
                        PrintProgram::dec_indent(PrintProgram::write_ln(
                            "".to_string(),
                            PrintProgram::inc_indent(PrintProgram::write_ln(
                                "else".to_string(),
                                print_program_from_statement(
                                    else_stmt,
                                    PrintProgram::dec_indent(k),
                                ),
                            )),
                        )),
                    ),
                )),
            ),
        ),
    }
}

fn print_program_from_while_do_statement<TNext>(
    wds: &WhileDoStatement,
    k: PrintProgram<TNext>,
) -> PrintProgram<TNext>
where
    TNext: Default,
{
    match wds {
        WhileDoStatement(expr, stmt) => PrintProgram::write(
            "while ".to_string(),
            print_program_from_expression(
                expr,
                PrintProgram::inc_indent(PrintProgram::write_ln(
                    " do".to_string(),
                    print_program_from_statement(stmt, PrintProgram::dec_indent(k)),
                )),
            ),
        ),
    }
}

fn print_program_from_expression_list<TNext>(
    el: &ExpressionList,
    k: PrintProgram<TNext>,
) -> PrintProgram<TNext>
where
    TNext: Default,
{
    let ExpressionList(NonEmptyVec(exprs)) = el;
    print_program_from_expression_slice(exprs, k)
}

fn print_program_from_expression_slice<TNext>(
    el: &[Expression],
    k: PrintProgram<TNext>,
) -> PrintProgram<TNext>
where
    TNext: Default,
{
    print_program_interpose(
        el,
        &print_program_from_expression,
        &|k| PrintProgram::write(", ".to_string(), k),
        k,
    )
}

fn print_program_from_expression<TNext>(
    el: &Expression,
    k: PrintProgram<TNext>,
) -> PrintProgram<TNext>
where
    TNext: Default,
{
    match el {
        Expression::Simple(se) => print_program_from_simple_expression(se.deref(), k),
        Expression::Relation(lhs, relop, rhs) => print_program_from_simple_expression(
            lhs.deref(),
            PrintProgram::write(
                format!(
                    " {} ",
                    match relop {
                        RelOp::Equal => "=",
                        RelOp::NotEqual => "<>",
                        RelOp::LessThan => "<",
                        RelOp::LessThanOrEqual => "<=",
                        RelOp::GreaterThan => ">",
                        RelOp::GreaterThanOrEqual => ">=",
                    }
                ),
                print_program_from_simple_expression(rhs, k),
            ),
        ),
    }
}

fn print_program_from_simple_expression<TNext>(
    se: &SimpleExpression,
    k: PrintProgram<TNext>,
) -> PrintProgram<TNext>
where
    TNext: Default,
{
    match se {
        SimpleExpression::Term(term) => print_program_from_term(term, k),
    }
}

fn print_program_from_term<TNext>(t: &Term, k: PrintProgram<TNext>) -> PrintProgram<TNext>
where
    TNext: Default,
{
    match t {
        Term::Factor(f) => print_program_from_factor(f, k),
        Term::MulOp(lhs, op, rhs) => print_program_from_factor(
            lhs.deref(),
            PrintProgram::write(
                format!(
                    " {} ",
                    match op {
                        MulOp::Star => "*",
                        MulOp::Slash => "/",
                        MulOp::Div => "div",
                        MulOp::Mod => "mod",
                        MulOp::And => "and",
                    }
                ),
                print_program_from_term(rhs.deref(), k),
            ),
        ),
    }
}

fn print_program_from_factor<TNext>(f: &Factor, k: PrintProgram<TNext>) -> PrintProgram<TNext>
where
    TNext: Default,
{
    match f {
        Factor::Id(id) => print_program_from_id(id, k),
        Factor::IdWithParams(_, _) => todo!(),
        Factor::Number(n) => PrintProgram::write(n.to_string(), k),
        Factor::Parens(_) => todo!(),
        Factor::Not(_) => todo!(),
        Factor::String(s) => print_program_from_string_literal(s, k),
    }
}

fn print_program_from_id<TNext>(id: &Id, k: PrintProgram<TNext>) -> PrintProgram<TNext>
where
    TNext: Default,
{
    PrintProgram::write(id.to_string(), k)
}

/// Print a string literal with apostrophes escaped.
/// For example, `foo'bar` becomes `'foo''bar'`.
fn print_program_from_string_literal<TNext>(
    s: &String,
    k: PrintProgram<TNext>,
) -> PrintProgram<TNext>
where
    TNext: Default,
{
    let espaced = s.replace("'", "''");
    PrintProgram::write(format!("'{}'", espaced).to_string(), k)
}

fn print_program_from_identifier_list<TNext>(
    il: &IdentifierList,
    k: PrintProgram<TNext>,
) -> PrintProgram<TNext>
where
    TNext: Default,
{
    let IdentifierList(NonEmptyVec(ids)) = il;
    print_program_from_id_slice(ids, k)
}

fn print_program_from_id_slice<TNext>(il: &[Id], k: PrintProgram<TNext>) -> PrintProgram<TNext>
where
    TNext: Default,
{
    print_program_interpose(
        il,
        &print_program_from_id,
        &|k| PrintProgram::write(", ".to_string(), k),
        k,
    )
}

#[cfg(test)]
mod tests {
    use crate::examples;

    use super::*;

    #[test]
    fn pretty_print_string_with_apostrophes() {
        let p = ProgramExpr::new(
            Id::new_from_str("helloWorld").unwrap(),
            IdentifierList::new(NonEmptyVec::single(Id::new_from_str("output").unwrap())),
            DeclarationsExpr::empty(),
            SubprogramDeclarations::empty(),
            CompoundStatement::new(vec![Statement::procedure(ProcedureStatement::with_params(
                // TODO: consider using an enum for the built-in procedures like writeLn
                Id::new_from_str("writeLn").unwrap(),
                ExpressionList::new(
                    NonEmptyVec::new(vec![Expression::simple(SimpleExpression::term(
                        Term::factor(Factor::string("foo'bar".to_string())),
                    ))])
                    .unwrap(),
                ),
            ))]),
        );
        let actual = pretty_print(p, 2);
        let expected = r#"
program helloWorld(output);
begin
  writeLn('foo''bar')
end."#
            .to_string()
            .trim()
            .to_string();

        assert_eq!(expected, actual);
    }

    #[test]
    fn pretty_print_hello_world() {
        let p = examples::hello_world();
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

    #[test]
    fn pretty_print_fizzbuzz() {
        let p = examples::fizzbuzz();
        let actual = pretty_print(p, 2);
        // TODO: elaborate this, for now it is just a smoke test
        assert!(actual.contains("program fizzbuzz(output);"));
        assert!(actual.contains("var"));
        assert!(actual.contains("i : integer;"));
        assert!(actual.contains("'Fizz'"));
        assert!(actual.contains("'Buzz'"));
        assert!(actual.contains("'FizzBuzz'"));
        assert!(actual.contains("i := 1;"));
        assert!(actual.contains("while i <= 100 do"));
        assert!(actual.contains("if i mod 15 = 0 then"));
        assert!(actual.contains("if i mod 5 = 0 then"));
        assert!(actual.contains("if i mod 3 = 0 then"));
    }
}
