//! Module with example Pascal programs.
//!
//! Why is this not under examples?
//!
//! It is internal to the crate rather than under examples for because
//! - it is used by the tests
//! - it is useful for demonstration purposes

use crate::il::{
    CompoundStatement, DeclarationsExpr, Expression, ExpressionList, Factor, Id, IdentifierList,
    NonEmptyVec, ProcedureStatement, ProgramExpr, SimpleExpression, Statement,
    SubprogramDeclarations, Term,
};

const HELLO_WORLD_PAS: &'static str = r#"
            program helloWorld(output);
            begin
                writeLn('Hello, World!')
            end.
        "#;

/// Creates a program expression for the Hello, World! program.
pub(crate) fn hello_world() -> ProgramExpr {
    ProgramExpr::new(
        Id::new_from_str("helloWorld").unwrap(),
        IdentifierList::new(NonEmptyVec::single(Id::new_from_str("output").unwrap())),
        DeclarationsExpr::empty(),
        SubprogramDeclarations::empty(),
        CompoundStatement::new(vec![Statement::procedure(ProcedureStatement::with_params(
            // TODO: consider using an enum for the built-in procedures like writeLn
            Id::new_from_str("writeLn").unwrap(),
            ExpressionList::new(
                NonEmptyVec::new(vec![Expression::simple(SimpleExpression::term(
                    Term::factor(Factor::string("Hello, World!")),
                ))])
                .unwrap(),
            ),
        ))]),
    )
}
