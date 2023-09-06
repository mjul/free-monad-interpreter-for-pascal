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
#[derive(Debug)]
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
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, PartialEq)]
pub(crate) struct DeclarationsExpr(pub(crate) Vec<VarDeclaration>);

impl DeclarationsExpr {
    /// Create an empty declarations expression (no var declarations)
    pub fn empty() -> Self {
        Self(vec![])
    }
    /// Create a declarations expression with a list of var declarations
    pub fn new(vds: Vec<VarDeclaration>) -> Self {
        Self(vds)
    }
}

/// Variable declaration (one or more vars of a given type)
// TODO: add type
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct VarDeclaration(pub(crate) IdentifierList, pub(crate) Type);

impl VarDeclaration {
    pub(crate) fn new(ids: IdentifierList, ty: Type) -> Self {
        Self(ids, ty)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Type {
    StandardType(StandardType),
    // TODO: arrays
}

impl Type {
    pub(crate) fn standard(standard_type: StandardType) -> Self {
        Self::StandardType(standard_type)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum StandardType {
    Integer,
    Real,
}

#[derive(Debug)]
pub(crate) struct SubprogramDeclarations(pub(crate) Vec<SubprogramDeclaration>);

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

#[derive(Debug)]
pub(crate) struct SubprogramDeclaration {}

/// A compound statement consisting of a `begin` and `end` block containing zero or more statements.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CompoundStatement(pub(crate) Vec<Statement>);

impl CompoundStatement {
    pub(crate) fn new(optional_statements: Vec<Statement>) -> Self {
        Self(optional_statements)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Statement {
    Assignment(AssignmentStatement),
    Procedure(ProcedureStatement),
    Compound(CompoundStatement),
    IfThenElse(IfThenElseStatement),
    WhileDo(WhileDoStatement),
}

/// A statement:
/// - a variable assignment
/// - a procedure call
/// - a compound statement
/// - an `if` `then` `else` statement
/// - a `while` `do` statement
impl Statement {
    /// Variable assignment statement
    pub(crate) fn assignment(vas: AssignmentStatement) -> Statement {
        Statement::Assignment(vas)
    }
    pub(crate) fn procedure(ps: ProcedureStatement) -> Statement {
        Statement::Procedure(ps)
    }
    pub(crate) fn compound(cs: CompoundStatement) -> Statement {
        Statement::Compound(cs)
    }
    pub(crate) fn if_then_else(ites: IfThenElseStatement) -> Statement {
        Statement::IfThenElse(ites)
    }
    pub(crate) fn while_do(ws: WhileDoStatement) -> Statement {
        Statement::WhileDo(ws)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct AssignmentStatement(pub(crate) Variable, pub(crate) Expression);

impl AssignmentStatement {
    pub(crate) fn new(lvar: Variable, value: Expression) -> Self {
        Self(lvar, value)
    }
}

/// A variable, simplified: here just an identifier or an array index
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Variable {
    Id(Id),
    ArrayIndex(Id, Expression),
}

impl Variable {
    /// A simple named variable
    pub(crate) fn id(id: Id) -> Self {
        Self::Id(id)
    }
    /// A position in a named array
    pub(crate) fn array_index(id: Id, index: Expression) -> Self {
        Self::ArrayIndex(id, index)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ProcedureStatement(pub(crate) Id, pub(crate) Option<ExpressionList>);

impl ProcedureStatement {
    pub(crate) fn new(id: Id) -> Self {
        Self(id, None)
    }
    pub(crate) fn with_params(id: Id, params: ExpressionList) -> Self {
        Self(id, Some(params))
    }
}

/// An `if` `then` `else` statement
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct IfThenElseStatement(
    pub(crate) Box<Expression>,
    pub(crate) Box<Statement>,
    pub(crate) Box<Statement>,
);

impl IfThenElseStatement {
    pub(crate) fn new(expr: Expression, then_stmt: Statement, else_stmt: Statement) -> Self {
        Self(Box::new(expr), Box::new(then_stmt), Box::new(else_stmt))
    }
}

/// A `while` *expression* `do` statement
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct WhileDoStatement(pub(crate) Box<Expression>, pub(crate) Box<Statement>);

impl WhileDoStatement {
    pub(crate) fn new(expr: Expression, statement: Statement) -> Self {
        Self(Box::new(expr), Box::new(statement))
    }
}

#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ExpressionList(pub(crate) NonEmptyVec<Expression>);

impl ExpressionList {
    /// Create a new [ExpressionList] from a non-empty vector of [Expression]s
    pub(crate) fn new(exprs: NonEmptyVec<Expression>) -> Self {
        Self(exprs)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Expression {
    Simple(Box<SimpleExpression>),
    Relation(Box<SimpleExpression>, RelOp, Box<SimpleExpression>),
}

impl Expression {
    /// Create a simple expression
    pub(crate) fn simple(simple_expression: SimpleExpression) -> Self {
        Self::Simple(Box::new(simple_expression))
    }
    /// Create a relation expression, *e.g.* `a < b`
    pub(crate) fn relation(lhs: SimpleExpression, relation: RelOp, rhs: SimpleExpression) -> Self {
        Self::Relation(Box::new(lhs), relation, Box::new(rhs))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum RelOp {
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    // TODO: IN
}

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Term {
    Factor(Factor),
    MulOp(Box<Factor>, MulOp, Box<Term>),
}

impl Term {
    pub(crate) fn factor(factor: Factor) -> Self {
        Self::Factor(factor)
    }
    pub(crate) fn mul_op(rhs: Factor, mul_op: MulOp, lhs: Term) -> Self {
        Self::MulOp(Box::new(rhs), mul_op, Box::new(lhs))
    }
}

#[derive(Debug, Clone, PartialEq)]
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
    pub(crate) fn string(s: String) -> Self {
        Self::String(s)
    }
    pub(crate) fn id(id: Id) -> Self {
        Self::Id(id)
    }
    pub(crate) fn number(n: i32) -> Self {
        Self::Number(n)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Sign {
    Plus,
    Minus,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum AddOp {
    Plus,
    Minus,
    Or,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum MulOp {
    Star,
    Slash,
    Div,
    Mod,
    And,
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
#[derive(Debug, Clone, PartialEq)]
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
                        Term::factor(Factor::string("Hello, World!".to_string())),
                    ))])
                    .unwrap(),
                ),
            ))]),
        );
        // if we can compile this it is representable in the IL
        assert_eq!(true, true);
    }
}
