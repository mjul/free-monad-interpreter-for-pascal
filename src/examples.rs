//! Module with example Pascal programs.
//!
//! Why is this not under examples?
//!
//! It is internal to the crate rather than under examples for because
//! - it is used by the tests
//! - it is useful for demonstration purposes

use crate::il::{AssignmentStatement, CompoundStatement, DeclarationsExpr, Expression, ExpressionList, Factor, Id, IdentifierList, IfThenElseStatement, NonEmptyVec, ProcedureStatement, ProgramExpr, RelOp, SimpleExpression, StandardType, Statement, SubprogramDeclarations, Term, Type, VarDeclaration, Variable, WhileDoStatement};

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

// Since we don't have for loops in our Pascal dialect, we use a while loop.
const FIZZBUZZ_PAS: &'static str = r#"
        program fizzbuzz(output);

        var
          i: integer;

        begin
          i := 1;
          while i <= 100 do
            if i mod 15 = 0 then
              writeln('FizzBuzz')
            else if i mod 3 = 0 then
              writeln('Fizz')
            else if i mod 5 = 0 then
              writeln('Buzz')
            else
              writeln(i);
        end.
    "#;


/// Creates a program expression for the Hello, World! program.
pub(crate) fn fizzbuzz() -> ProgramExpr {
    let write_ln_term = |term|
        Statement::procedure(ProcedureStatement::with_params(
            Id::new_from_str("writeLn").unwrap(),
            ExpressionList::new(
                NonEmptyVec::new(vec![Expression::simple(SimpleExpression::term(
                    term,
                ))])
                    .unwrap(),
            ),
        ));
    let write_ln_str = |str|
        write_ln_term(Term::factor(Factor::string(str)));
    let assign_int = |id, value|
        Statement::assignment(
            AssignmentStatement::new(
                Variable::id(Id::new_from_str(id).unwrap()),
                Expression::simple(
                    SimpleExpression::term(Term::factor(Factor::number(value))))));

    let while_leq = |id, limit_inclusive, body|
        Statement::while_do(
            WhileDoStatement::new(
                Expression::relation(
                    SimpleExpression::term(Term::factor(Factor::id(Id::new_from_str(id).unwrap()))),
                    RelOp::LessThanOrEqual,
                    SimpleExpression::term(Term::factor(Factor::number(limit_inclusive)))),
                body));
    let if_then_else = |cond, then_stmt, else_stmt|
        Statement::if_then_else(
            IfThenElseStatement::new(cond, then_stmt, else_stmt));
    let mod_eq = |id, divisor, remainder|
        Expression::relation(
            SimpleExpression::term(Term::factor(Factor::id(Id::new_from_str(id).unwrap()))),
            RelOp::Equal,
            SimpleExpression::term(Term::factor(Factor::number(remainder))));

    ProgramExpr::new(
        Id::new_from_str("fizzbuzz").unwrap(),
        IdentifierList::new(NonEmptyVec::single(Id::new_from_str("output").unwrap())),
        DeclarationsExpr::new(
            vec![VarDeclaration::new(
                IdentifierList::new(NonEmptyVec::new(vec![Id::new_from_str("i").unwrap()]).unwrap()),
                Type::standard(StandardType::Integer))]),
        SubprogramDeclarations::empty(),
        CompoundStatement::new(
            vec![
                assign_int("i", 1),
                while_leq("i", 100,
                    if_then_else(
                        mod_eq("i", 15, 0),
                          write_ln_str("FizzBuzz"),
                        if_then_else(
                            mod_eq("i", 3, 0),
                            write_ln_str("Fizz"),
                            if_then_else(
                                mod_eq("i", 5, 0),
                                write_ln_str("Buzz"),
                                write_ln_term(Term::factor(Factor::id(Id::new_from_str("i").unwrap())))

                )))),
            ]),
    )
}