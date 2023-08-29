//! Intermediate language representing the Pascal program as an abstract syntax tree.

use std::error::Error;
use std::fmt::{Debug, Display, Formatter};


/// Pascal language expressions, everything between `program` and `end.`
pub(crate) enum PascalExpr {
    Program(ProgramExpr),
    IdentifierList(IdentifierList),
    Declarations(DeclarationsExpr),
    SubprogramDeclarations(SubprogramDeclarations),
    CompoundStatement(CompoundStatement),
}


/// Pascal Program Expression, `program`  ... `.`
///
/// Example Pascal Program:
/// ```pascal
/// program helloWorld(output);
/// begin
///     writeLn('Hello, World!')
/// end.
/// ```
pub(crate) struct ProgramExpr {
    pub(crate) id: Id,
    pub(crate) identifier_list: IdentifierList,
    pub(crate) declarations: DeclarationsExpr,
    pub(crate) subprogram_declarations: SubprogramDeclarations,
    pub(crate) compound_statement: CompoundStatement,
}

impl ProgramExpr {
    pub fn new(
        id: Id,
        identifier_list: IdentifierList,
        declarations: DeclarationsExpr,
        subprogram_declarations: SubprogramDeclarations,
        compound_statement: CompoundStatement,
    ) -> Self {
        Self {
            id,
            identifier_list,
            declarations,
            subprogram_declarations,
            compound_statement,
        }
    }
}

/// Non-empty list of identifiers
#[derive(Debug, Clone)]
pub(crate) struct IdentifierList(pub(crate) NonEmptyVec<Id>);

impl IdentifierList {
    fn new_single(id: Id) -> Self {
        Self::new(NonEmptyVec::single(id))
    }
    pub(crate) fn new(identifiers: NonEmptyVec<Id>) -> Self {
        Self(identifiers)
    }
}

// TODO: elaborate on this
pub(crate) struct DeclarationsExpr();

impl DeclarationsExpr {
    pub fn empty() -> Self {
        Self {}
    }
}

pub(crate) struct SubprogramDeclarations(Vec<SubprogramDeclaration>);

impl SubprogramDeclarations {
    /// Create an empty [SubprogramDeclarations]
    pub fn empty() -> Self {
        Self(vec![])
    }
    /// Create a new [SubprogramDeclarations] from a vector of [SubprogramDeclaration]s
    pub fn new(xs: Vec<SubprogramDeclaration>) -> Self {
        Self(xs)
    }
}

pub(crate) struct SubprogramDeclaration {}

/// A compound statement consisting of a `begin` and `end` block containing zero or more statements.
pub(crate) struct CompoundStatement(Vec<Statement>);

impl CompoundStatement {
    pub(crate) fn new(optional_statements: Vec<Statement>) -> Self {
        Self(optional_statements)
    }
}

pub(crate) enum Statement {
    // TODO: variable assignment
    Procedure(ProcedureStatement),
    Compound(CompoundStatement),
    // TODO: if then else
    // TODO: while do
}

/// A statement:
/// - a variable assignment
/// - a procedure call
/// - a compound statement
/// - an `if` `then` `else` statement
/// - a `while` `do` statement
impl Statement {
    pub(crate) fn procedure(ps: ProcedureStatement) -> Statement {
        Statement::Procedure(ps)
    }
    pub(crate) fn compound(cs: CompoundStatement) -> Statement {
        Statement::Compound(cs)
    }
}

pub(crate) struct ProcedureStatement(Id, Option<ExpressionList>);

impl ProcedureStatement {
    pub(crate) fn new(id: Id) -> Self {
        Self(id, None)
    }
    pub(crate) fn with_params(id: Id, params: ExpressionList) -> Self {
        Self(id, Some(params))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct NonEmptyVec<T>(pub(crate) Vec<T>);

impl<T> NonEmptyVec<T> {
    /// Create a non-empty vector containing a single item.
    pub(crate) fn single(item: T) -> NonEmptyVec<T> {
        NonEmptyVec(vec![item])
    }

    // TODO: a proper error type
    pub(crate) fn new(xs: Vec<T>) -> Result<Self, ()> {
        match xs.len() {
            0 => Err(()),
            _ => Ok(Self(xs)),
        }
    }
}

/// A non-empty list of expressions
pub(crate) struct ExpressionList(NonEmptyVec<Expression>);

impl ExpressionList {
    /// Create a new [ExpressionList] from a non-empty vector of [Expression]s
    pub(crate) fn new(exprs: NonEmptyVec<Expression>) -> Self {
        Self(exprs)
    }
}

pub(crate) enum Expression {
    Simple(Box<SimpleExpression>),
    Relation(Box<SimpleExpression>, RelOp, Box<SimpleExpression>),
}

impl Expression {
    pub(crate) fn simple(simple_expression: SimpleExpression) -> Self {
        Self::Simple(Box::new(simple_expression))
    }
}

pub(crate) enum RelOp {
    Equal,
    NotEqual,
    // TODO: elaborate < > <= >=
}

pub(crate) enum SimpleExpression {
    Term(Term),
    // TODO: SignTerm(Sign, Term),
    // TODO: AddTerm(SimpleExpression, AddOp, Term),
}

impl SimpleExpression {
    pub(crate) fn term(term: Term) -> Self {
        Self::Term(term)
    }
}

pub(crate) enum Term {
    Factor(Factor),
}

impl Term {
    pub(crate) fn factor(factor: Factor) -> Self {
        Self::Factor(factor)
    }
}

pub(crate) enum Factor {
    Id(Id),
    IdWithParams(Id, ExpressionList),
    Number(i32),
    /// `(` expression `)`
    Parens(Expression),
    Not(Box<Factor>),
    // The string factor is part of the Pascal language, but not part of Tiny Pascal
    String(String),
}

impl Factor {
    pub(crate) fn string(s: &str) -> Self {
        Self::String(String::from(s))
    }
}

pub(crate) enum Sign {
    Plus,
    Minus,
}

pub(crate) enum AddOp {
    Plus,
    Minus,
    Or,
}

/// Error representing an invalid tokens
#[derive(Debug)]
pub(crate) enum TokenError {
    InvalidIdentifier(String),
}

impl Display for TokenError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl Error for TokenError {}

/// Identifier
#[derive(Debug,Clone)]
pub(crate) struct Id(String);

impl Id {
    pub(crate) fn new_from_str(id: &str) -> Result<Id, TokenError> {
        let starts_with_letter = id
            .chars()
            .next()
            .map(|c| c.is_ascii_alphabetic())
            .unwrap_or(false);
        let rest_is_alphanumeric_or_underscore =
            id.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
        let is_valid = starts_with_letter && rest_is_alphanumeric_or_underscore;
        match is_valid {
            true => Ok(Self(id.to_string())),
            false => Err(TokenError::InvalidIdentifier(id.to_string())),
        }
    }
}

impl Display for Id {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    const HELLO_WORLD_PAS: &'static str = r#"
            program helloWorld(output);
            begin
                writeLn('Hello, World!')
            end.
        "#;

    #[test]
    fn program_expr_can_represent_hello_world() {
        let hello_world = ProgramExpr::new(
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
        );
        // if we can compile this it is representable in the IL
        assert_eq!(true, true);
    }
}
