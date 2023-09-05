//! Front end for the Pascal interpreter.
//! This module parses Pascal files to the intermediate representation.
//!
//! It uses the Pest parser generator to parse the Pascal files,
//! see <https://pest.rs/>

use std::fmt::{Display, Formatter};
use pest::iterators::Pair;
use pest::Parser;
use pest_derive::Parser;

use crate::il;
use crate::il::{DeclarationsExpr, StandardType, Type};

#[derive(Parser)]
#[grammar = "src/front_end/pascal_grammar.pest"]
pub struct PascalParser;

#[derive(Debug)]
pub(crate) enum FrontEndError {
    ParseError(pest::error::Error<Rule>),
    ConversionError(ConversionError),
}

impl Display for FrontEndError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            FrontEndError::ParseError(e) => write!(f, "Parse error: {}", e),
            FrontEndError::ConversionError(e) => write!(f, "Conversion error: {}", e),
        }
    }
}

impl std::error::Error for FrontEndError {
}

#[derive(Debug)]
pub(crate) enum ConversionError {
    ParseResultNotSingular,
    UnexpectedRuleInPair(Rule),
    ConversionError(String),
}
impl Display for ConversionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ConversionError::ParseResultNotSingular => {
                write!(f, "Expected a single parse result")
            }
            ConversionError::UnexpectedRuleInPair(rule) => {
                write!(f, "Unexpected rule in pair: {:?}", rule)
            }
            ConversionError::ConversionError(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for ConversionError {

}

fn il_id_from(pair: &Pair<Rule>) -> Result<il::Id, ConversionError> {
    match &pair.as_rule() {
        Rule::IDENT => {
            let id = il::Id::new_from_str(pair.as_str()).unwrap();
            Ok(id)
        }
        _ => Err(ConversionError::UnexpectedRuleInPair(pair.as_rule())),
    }
}

fn il_identifier_list_from(pair: &Pair<Rule>) -> Result<il::IdentifierList, ConversionError> {
    match &pair.as_rule() {
        Rule::identifier_list => {
            let ids = pair
                .clone()
                .into_inner()
                .filter_map(|p| match p.as_rule() {
                    Rule::IDENT => Some(il_id_from(&p)),
                    Rule::COMMA => None,
                    _ => Some(Err(ConversionError::UnexpectedRuleInPair(p.as_rule()))),
                })
                .collect::<Result<Vec<il::Id>, ConversionError>>()?;
            let idv = il::NonEmptyVec::new(ids.to_vec()).map_err(|_| {
                ConversionError::ConversionError(format!(
                    "Failed to create NonEmptyVec from identifier_list: {:?}",
                    &pair
                ))
            })?;
            Ok(il::IdentifierList::new(idv))
        }
        _ => Err(ConversionError::UnexpectedRuleInPair(pair.as_rule())),
    }
}

fn il_declarations_from(pair: &Pair<Rule>) -> Result<il::DeclarationsExpr, ConversionError> {
    match &pair.as_rule() {
        Rule::declarations => {
            // Chunking in 5 because a single declaration is:
            // var, identifier_list, colon, type, semicolon
            let vds = pair
                .clone()
                .into_inner()
                .into_iter()
                .collect::<Vec<Pair<Rule>>>()
                .chunks(5)
                .filter_map(|chunk| match &chunk[..] {
                    [] => None,
                    [var, il, colon, tp, semicolon] => {
                        match (
                            &var.as_rule(),
                            &il.as_rule(),
                            &colon.as_rule(),
                            &tp.as_rule(),
                            &semicolon.as_rule(),
                        ) {
                            (
                                Rule::VAR,
                                Rule::identifier_list,
                                Rule::COLON,
                                Rule::r#type,
                                Rule::SEMICOLON,
                            ) => {
                                let ids = il_identifier_list_from(il).map_err(|e| Some(e));
                                let ty = il_type_from(tp).map_err(|e| Some(e));
                                match (ids, ty) {
                                    (Ok(ids), Ok(ty)) => {
                                        let vd = il::VarDeclaration::new(ids, ty);
                                        Some(Ok(vd))
                                    }
                                    (Err(e), _) => Some(Err(e.unwrap())),
                                    (_, Err(e)) => Some(Err(e.unwrap())),
                                }
                            }
                            _ => Some(Err(ConversionError::ConversionError(format!(
                                "Unexpected rule in chunk for var declaration {:?}",
                                chunk
                            )))),
                        }
                    }
                    _ => Some(Err(ConversionError::ConversionError(format!(
                        "Unexpected number of pairs in chunk for var declaration {:?}",
                        chunk
                    )))),
                })
                .collect::<Result<Vec<il::VarDeclaration>, ConversionError>>()?;
            Ok(DeclarationsExpr::new(vds))
        }

        _ => Err(ConversionError::UnexpectedRuleInPair(pair.as_rule())),
    }
}

fn il_type_from(pair: &Pair<Rule>) -> Result<il::Type, ConversionError> {
    match &pair.as_rule() {
        Rule::r#type => {
            let inners = pair.clone().into_inner().collect::<Vec<Pair<Rule>>>();
            match &inners[..] {
                [p] => match p.as_rule() {
                    Rule::standard_type => il_standard_type_from(&p).map(|st| Type::standard(st)),
                    _ => Err(ConversionError::UnexpectedRuleInPair(pair.as_rule())),
                },
                _ => todo!(),
            }
        }
        _ => Err(ConversionError::UnexpectedRuleInPair(pair.as_rule())),
    }
}

fn il_standard_type_from(pair: &Pair<Rule>) -> Result<StandardType, ConversionError> {
    match &pair.as_rule() {
        Rule::standard_type => {
            let inners: Vec<Pair<Rule>> = pair.clone().into_inner().into_iter().collect();
            match &inners[..] {
                [p] => match p.as_rule() {
                    Rule::INTEGER => Ok(StandardType::Integer),
                    Rule::REAL => Ok(StandardType::Real),
                    _ => Err(ConversionError::UnexpectedRuleInPair(pair.as_rule())),
                },
                _ => Err(ConversionError::ConversionError(
                    "Unexpected number of pairs under standard_type rule".to_string(),
                )),
            }
        }
        _ => Err(ConversionError::UnexpectedRuleInPair(pair.as_rule())),
    }
}

fn il_subprogram_declarations_from(
    pair: &Pair<Rule>,
) -> Result<il::SubprogramDeclarations, ConversionError> {
    match &pair.as_rule() {
        Rule::subprogram_declarations => match pair.clone().into_inner().next() {
            None => Ok(il::SubprogramDeclarations::empty()),
            Some(p) => todo!(),
        },
        _ => Err(ConversionError::UnexpectedRuleInPair(pair.as_rule())),
    }
}

fn il_compound_statement_from(pair: &Pair<Rule>) -> Result<il::CompoundStatement, ConversionError> {
    match &pair.as_rule() {
        Rule::compound_statement => {
            let inners: Vec<Pair<Rule>> = pair.clone().into_inner().into_iter().collect();
            match &inners[..] {
                [_begin, optional_statements, _end] => {
                    let stmts = il_statement_list_from_optional_statements(optional_statements)?;
                    Ok(il::CompoundStatement::new(stmts))
                }
                _ => Err(ConversionError::ConversionError(
                    "Unexpected number of pairs under compound_statement rule".to_string(),
                )),
            }
        }
        _ => Err(ConversionError::UnexpectedRuleInPair(pair.as_rule())),
    }
}
/*
                   let stmts_vec =
                       optional_statements
                           .clone()
                           .into_inner()
                           .into_iter()
                           .map(|p| il_statement_list_from(&p))
                           .collect::<Result<Vec<Vec<il::Statement>>, ConversionError>>()?;
                   match stmts_vec.len() {
                       0 => Ok(il::CompoundStatement::new(vec![])),
                       1 => Ok(il::CompoundStatement::new (stmts_vec[0].to_vec())),
                       _ => Err(ConversionError::ConversionError("Unexpected number of pairs in optional_statement rule".to_string())),
                   }

*/
fn il_statement_list_from_optional_statements(
    pair: &Pair<Rule>,
) -> Result<Vec<il::Statement>, ConversionError> {
    match &pair.as_rule() {
        Rule::optional_statements => {
            let inners: Vec<Pair<Rule>> = pair.clone().into_inner().into_iter().collect();
            match &inners[..] {
                [statement_list] => il_statement_list_from(statement_list),
                [] => Ok(vec![]),
                _ => Err(ConversionError::ConversionError(
                    "Unexpected number of pairs under optional_statements rule".to_string(),
                )),
            }
        }
        _ => Err(ConversionError::UnexpectedRuleInPair(pair.as_rule())),
    }
}

fn il_statement_list_from(pair: &Pair<Rule>) -> Result<Vec<il::Statement>, ConversionError> {
    match &pair.as_rule() {
        Rule::statement_list => {
            let stmts = pair
                .clone()
                .into_inner()
                .filter_map(|p| match p.as_rule() {
                    Rule::statement => Some(il_statement_from(&p)),
                    Rule::SEMICOLON => None,
                    _ => Some(Err(ConversionError::UnexpectedRuleInPair(p.as_rule()))),
                })
                .collect::<Result<Vec<il::Statement>, ConversionError>>()?;
            Ok(stmts)
        }
        _ => Err(ConversionError::UnexpectedRuleInPair(pair.as_rule())),
    }
}

fn il_statement_from(pair: &Pair<Rule>) -> Result<il::Statement, ConversionError> {
    match pair.as_rule() {
        Rule::statement => {
            let inners: Vec<Pair<Rule>> = pair.clone().into_inner().into_iter().collect();
            match inners.first() {
                None => Err(ConversionError::ConversionError(
                    "Unexpected empty statement".to_string(),
                )),
                Some(first_inner) => match first_inner.as_rule() {
                    Rule::procedure_statement => {
                        let ps = il_procedure_statement_from(first_inner)?;
                        Ok(il::Statement::procedure(ps))
                    }
                    Rule::variable => match &inners[..] {
                        [variable, _assign, expression] => {
                            let v = il_variable_from(variable)?;
                            let e = il_expression_from(expression)?;
                            Ok(il::Statement::assignment(il::AssignmentStatement::new(
                                v, e,
                            )))
                        }
                        _ => todo!(),
                    },
                    Rule::WHILE => match &inners[..] {
                        [_while, expr, _do, stmt] => {
                            let e = il_expression_from(expr)?;
                            let s = il_statement_from(stmt)?;
                            Ok(il::Statement::while_do(il::WhileDoStatement::new(e, s)))
                        }
                        _ => todo!("Unexpected number of pairs in WHILE rule: {:?}", inners),
                    },
                    Rule::IF => match &inners[..] {
                        [_if, expr, _then, thn, _else, els] => {
                            let cond_expr = il_expression_from(expr)?;
                            let then_stmt = il_statement_from(thn)?;
                            let else_stmt = il_statement_from(els)?;
                            Ok(il::Statement::if_then_else(il::IfThenElseStatement::new(
                                cond_expr, then_stmt, else_stmt,
                            )))
                        }
                        _ => todo!("Unexpected number of pairs in IF rule: {:?}", inners),
                    },
                    _ => todo!("il_statement_from inner: {:?}", first_inner.as_rule()),
                },
            }
        }
        _ => Err(ConversionError::UnexpectedRuleInPair(pair.as_rule())),
    }
}

fn il_variable_from(pair: &Pair<Rule>) -> Result<il::Variable, ConversionError> {
    match &pair.as_rule() {
        Rule::variable => {
            let inners: Vec<Pair<Rule>> = pair.clone().into_inner().into_iter().collect();
            match inners.len() {
                0 => Err(ConversionError::ConversionError(
                    "Unexpected empty variable".to_string(),
                )),
                1 => {
                    let id = il_id_from(&inners[0])?;
                    Ok(il::Variable::id(id))
                }
                _ => todo!("il_variable_from: {:?}", inners),
            }
        }
        _ => Err(ConversionError::UnexpectedRuleInPair(pair.as_rule())),
    }
}

fn il_procedure_statement_from(
    pair: &Pair<Rule>,
) -> Result<il::ProcedureStatement, ConversionError> {
    match &pair.as_rule() {
        Rule::procedure_statement => {
            let inners: Vec<Pair<Rule>> = pair.clone().into_inner().into_iter().collect();
            match &inners[..] {
                [id, _lparen, expression_list, _rparen] => {
                    let id = il_id_from(&id)?;
                    let expression_list = il_expression_list_from(&expression_list)?;
                    Ok(il::ProcedureStatement::with_params(id, expression_list))
                }
                [id] => {
                    let id = il_id_from(&id)?;
                    Ok(il::ProcedureStatement::new(id))
                }
                _ => Err(ConversionError::ConversionError(
                    "Unexpected number of pairs under procedure_statement rule".to_string(),
                )),
            }
        }
        _ => Err(ConversionError::UnexpectedRuleInPair(pair.as_rule())),
    }
}

fn il_expression_list_from(pair: &Pair<Rule>) -> Result<il::ExpressionList, ConversionError> {
    match &pair.as_rule() {
        Rule::expression_list => {
            let exprs = &pair
                .clone()
                .into_inner()
                .into_iter()
                .filter_map(|p| match p.as_rule() {
                    Rule::expression => Some(il_expression_from(&p)),
                    Rule::COMMA => None,
                    _ => Some(Err(ConversionError::UnexpectedRuleInPair(p.as_rule()))),
                })
                .collect::<Result<Vec<il::Expression>, ConversionError>>()?;
            let ne_exprs = il::NonEmptyVec::new(exprs.to_vec()).map_err(|_| {
                ConversionError::ConversionError(
                    "Failed to create NonEmptyVec from expression_list".to_string(),
                )
            })?;
            Ok(il::ExpressionList::new(ne_exprs))
        }
        _ => Err(ConversionError::UnexpectedRuleInPair(pair.as_rule())),
    }
}

fn il_expression_from(pair: &Pair<Rule>) -> Result<il::Expression, ConversionError> {
    match &pair.as_rule() {
        Rule::expression => {
            let inners: Vec<Pair<Rule>> = pair.clone().into_inner().into_iter().collect();
            match &inners[..] {
                [] => Err(ConversionError::ConversionError(
                    "Unexpected empty expression".to_string(),
                )),
                [se_pair] => {
                    let se = il_simple_expression_from(se_pair)?;
                    Ok(il::Expression::simple(se))
                }
                [lhs_pair, rel_pair, rhs_pair] => {
                    let lhs = il_simple_expression_from(lhs_pair)?;
                    let rel = il_relop_from(rel_pair)?;
                    let rhs = il_simple_expression_from(rhs_pair)?;
                    Ok(il::Expression::relation(lhs, rel, rhs))
                }
                _ => todo!("il_expression_from: {:?}", inners),
            }
        }
        _ => Err(ConversionError::UnexpectedRuleInPair(pair.as_rule())),
    }
}

fn il_relop_from(pair: &Pair<Rule>) -> Result<il::RelOp, ConversionError> {
    match &pair.as_rule() {
        Rule::RELOP => {
            let inner_rules: Vec<Rule> = pair
                .clone()
                .into_inner()
                .into_iter()
                .map(|p| p.as_rule())
                .collect();
            match &inner_rules[..] {
                [Rule::EQUAL] => Ok(il::RelOp::Equal),
                [Rule::NOT_EQUAL] => Ok(il::RelOp::NotEqual),
                [Rule::LT] => Ok(il::RelOp::LessThan),
                [Rule::LE] => Ok(il::RelOp::LessThanOrEqual),
                [Rule::GT] => Ok(il::RelOp::GreaterThan),
                [Rule::GE] => Ok(il::RelOp::GreaterThanOrEqual),
                _ => Err(ConversionError::ConversionError(format!(
                    "Unexpected relop: {}",
                    pair.as_str()
                ))),
            }
        }
        _ => Err(ConversionError::UnexpectedRuleInPair(pair.as_rule())),
    }
}

fn il_simple_expression_from(pair: &Pair<Rule>) -> Result<il::SimpleExpression, ConversionError> {
    match &pair.as_rule() {
        Rule::simple_expression => {
            // Let's try a different style and step through the inners with .next()
            // Unfortunately we still need to clone since we only have a borrowed pair
            let mut inners = pair.clone().into_inner();
            let first_inner = inners.next().ok_or(ConversionError::ConversionError(
                "Unexpected empty simple_expression".to_string(),
            ))?;
            match &first_inner.as_rule() {
                Rule::term => {
                    // term
                    let t = il_term_from(&first_inner)?;
                    Ok(il::SimpleExpression::term(t))
                }
                Rule::sign => {
                    // sign term
                    let s = il_sign_from(&first_inner)?;
                    let next2 = inners.next().ok_or(ConversionError::ConversionError(
                        "Missing term after sign".to_string(),
                    ))?;
                    let t = il_term_from(&next2)?;
                    todo!("il_simple_expression_from: sign term: {:?}", s);
                }
                _ => Err(ConversionError::UnexpectedRuleInPair(pair.as_rule())),
            }
        }
        _ => Err(ConversionError::UnexpectedRuleInPair(pair.as_rule())),
    }
}

fn il_term_from(pair: &Pair<Rule>) -> Result<il::Term, ConversionError> {
    match &pair.as_rule() {
        Rule::term => {
            let inners: Vec<Pair<Rule>> = pair.clone().into_inner().into_iter().collect();
            match &inners[..] {
                [] => Err(ConversionError::ConversionError(
                    "Unexpected empty term".to_string(),
                )),
                [factor] => {
                    let f = il_factor_from(factor)?;
                    Ok(il::Term::factor(f))
                }
                [factor, mulop, term] => {
                    let f = il_factor_from(factor)?;
                    let m = il_mulop_from(mulop)?;
                    let t = il_term_from(term)?;
                    Ok(il::Term::mul_op(f, m, t))
                }
                _ => todo!("il_term_from: {:?}", inners),
            }
        }
        _ => Err(ConversionError::UnexpectedRuleInPair(pair.as_rule())),
    }
}

fn il_mulop_from(pair: &Pair<Rule>) -> Result<il::MulOp, ConversionError> {
    match &pair.as_rule() {
        Rule::MULOP => match pair.clone().into_inner().next().map(|p| p.as_rule()) {
            Some(Rule::STAR) => Ok(il::MulOp::Star),
            Some(Rule::SLASH) => Ok(il::MulOp::Slash),
            Some(Rule::DIV) => Ok(il::MulOp::Div),
            Some(Rule::MOD) => Ok(il::MulOp::Mod),
            Some(Rule::AND) => Ok(il::MulOp::And),
            _ => Err(ConversionError::ConversionError(format!(
                "Unexpected mulop: {}",
                pair.as_str()
            ))),
        },
        _ => Err(ConversionError::UnexpectedRuleInPair(pair.as_rule())),
    }
}

fn il_factor_from(pair: &Pair<Rule>) -> Result<il::Factor, ConversionError> {
    match &pair.as_rule() {
        Rule::factor => {
            let inners = pair
                .clone()
                .into_inner()
                .into_iter()
                .collect::<Vec<Pair<Rule>>>();
            match &inners[..] {
                [p] => {
                    match p.as_rule() {
                        Rule::IDENT => {
                            let id = il_id_from(&p)?;
                            Ok(il::Factor::id(id))
                        }
                        Rule::unsigned_number => {
                            //let n = il_unsigned_number_from(&p)?;
                            //Ok(il::Factor::number(n))
                            todo!()
                        }
                        Rule::unsigned_constant => Ok(il::Factor::string(p.as_str())),
                        _ => Err(ConversionError::UnexpectedRuleInPair(p.as_rule())),
                    }
                }
                _ => todo!("il_factor_from: {:?}", inners),
            }
        }
        _ => Err(ConversionError::UnexpectedRuleInPair(pair.as_rule())),
    }
}

fn il_sign_from(pair: &Pair<Rule>) -> Result<il::Sign, ConversionError> {
    match &pair.as_rule() {
        Rule::sign => match pair.clone().into_inner().next().map(|p| p.as_rule()) {
            Some(Rule::PLUS) => Ok(il::Sign::Plus),
            Some(Rule::MINUS) => Ok(il::Sign::Minus),
            _ => Err(ConversionError::ConversionError(format!(
                "Unexpected sign: {}",
                pair.as_str()
            ))),
        },
        _ => Err(ConversionError::UnexpectedRuleInPair(pair.as_rule())),
    }
}

fn il_program_from(pair: Pair<Rule>) -> Result<il::ProgramExpr, ConversionError> {
    match &pair.as_rule() {
        Rule::program => {
            let inners: Vec<Pair<Rule>> = pair.into_inner().into_iter().collect();
            match &inners[..] {
                [_program, id, _lparen, identifier_list, _rparen, _semicolon, declarations, subprogram_declarations, compound_statement, ..] =>
                {
                    let id = il_id_from(&id)?;
                    let identifier_list = il_identifier_list_from(identifier_list)?;
                    let declarations = il_declarations_from(declarations)?;
                    let subprogram_declarations =
                        il_subprogram_declarations_from(subprogram_declarations)?;
                    let compound_statement = il_compound_statement_from(compound_statement)?;
                    Ok(il::ProgramExpr::new(
                        id,
                        identifier_list,
                        declarations,
                        subprogram_declarations,
                        compound_statement,
                    ))
                }
                _ => Err(ConversionError::ConversionError(
                    "Unexpected number of pairs under program rule".to_string(),
                )),
            }
        }
        _ => Err(ConversionError::UnexpectedRuleInPair(pair.as_rule())),
    }
}

/// Parse a Pascal program from its source code as a string.
/// This is the main entry point for the front-end.
pub(crate) fn parse_program_string(input: &str) -> Result<il::ProgramExpr, FrontEndError> {
    let result_pairs =
        PascalParser::parse(Rule::program, input).map_err(|e| FrontEndError::ParseError(e))?;
    match result_pairs.len() {
        1 => {
            let pair = result_pairs
                .into_iter()
                .next()
                .expect("Expected a parse result when we know there is one");
            let result: il::ProgramExpr =
                il_program_from(pair).map_err(|e| FrontEndError::ConversionError(e))?;
            Ok(result)
        }
        _ => Err(FrontEndError::ConversionError(
            ConversionError::ParseResultNotSingular,
        )),
    }
}

#[cfg(test)]
mod tests {
    use paste::paste;
    use pest::iterators::Pairs;

    use crate::il::{DeclarationsExpr, VarDeclaration};

    use super::*;

    #[test]
    fn pascal_parser_can_parse_whitespace_multiple_without_err() {
        let result_pairs = PascalParser::parse(Rule::WHITESPACE, "  \t\r\n");
        assert!(result_pairs.is_ok());
        assert_eq!(
            0,
            result_pairs.unwrap().count(),
            "Expected whitespace to parse to no tokens"
        );
    }

    fn assert_parse_consumed_all_input(pairs: Pairs<Rule>, rule: &Rule, input: &str) {
        assert_eq!(1, pairs.len());
        let p = pairs.into_iter().next().unwrap();
        assert_eq!(rule, &p.as_rule(), "Expected to parse rule {:?}", rule);
        assert_eq!(input, p.as_span().as_str(), "Expected to consume all input");
    }

    macro_rules! test_can_all {
        ($rule:ident, $case:ident, $input:expr) => {
            paste! {
                #[test]
                fn [<pascal_parser_can_parse_ $rule _ $case _without_err>]() {
                    let result_pairs = PascalParser::parse(Rule::$rule, $input);
                    dbg!(&result_pairs);
                    assert!(result_pairs.is_ok());
                    assert_parse_consumed_all_input(result_pairs.unwrap(), &Rule::$rule, $input);
                }
            }
        };
    }

    test_can_all!(IDENT,mixed_case, "helloWorld");

    test_can_all!(STRING_LITERAL, empty, "''");
    test_can_all!(STRING_LITERAL, simple, "'abc'");
    test_can_all!(STRING_LITERAL, with_quoted_quote, "'foo''bar'");

    test_can_all!(PROGRAM, lower_case, "program");
    test_can_all!(PROGRAM, mixed_case, "pRoGrAm");

    test_can_all!(identifier_list, single_id, "output");
    test_can_all!(identifier_list, multiple_ids, "a,b,c");

    test_can_all!(declarations, empty, "");
    test_can_all!(declarations, single_decl_single_var, "var a : integer;");
    test_can_all!(declarations, single_dec_multiple_vars, "var i, j : integer;");
    test_can_all!(declarations, multiple_decls, "var a : integer; var b: integer;");

    test_can_all!(subprogram_declarations, empty, "");
    test_can_all!(subprogram_declarations, single_function, "function foo: integer; begin end;");

    test_can_all!(subprogram_head, function_no_args, "function foo: integer;");
    test_can_all!(subprogram_head, function_with_args, "function foo(x:integer): integer;");
    test_can_all!(subprogram_head, procedure_no_args, "procedure foo;");
    test_can_all!(subprogram_head, procedure_with_args, "procedure foo(x:integer);");

    test_can_all!(arguments,_empty, "");
    test_can_all!(arguments, single, "(x:integer)");
    test_can_all!(arguments, multiple, "(x:integer; y : integer)");

    test_can_all!(parameter_list, single, "x:integer");
    test_can_all!(parameter_list, multiple_2, "x:integer; y : integer");
    test_can_all!(parameter_list, multiple_3, "x:integer; y : integer; z : integer");

    test_can_all!(BEGIN, lower_case, "begin");
    test_can_all!(END, lower_case, "end");

    test_can_all!(optional_statements, empty, "");
    test_can_all!(optional_statements, single_statement,  "x := 1");
    test_can_all!(optional_statements, multiple_statements, "x:=1; y:=2");

    test_can_all!(compound_statement, empty, "begin end");
    test_can_all!(compound_statement, single_assignment, "begin x:=1 end");
    test_can_all!(compound_statement, single_writeln, "begin writeLn('Hello, World!') end");
    test_can_all!(compound_statement, multiple, "begin x:=1; if x>10 then x:=10 else x:=x end");

    test_can_all!(statement, assignment, "x:=1");
    test_can_all!(statement, procedure_statement, "writeLn('Hello, World!')");
    test_can_all!(statement, compound_statement, "begin x:=1;y:=2 end");
    test_can_all!(statement, if_then_else, "if x>10 then x:=10 else x:=x");
    test_can_all!(statement, if_then_else_composite_cond, "if i mod 3 = 0 then x else y");
    test_can_all!(statement, while_less_than, "while x<10 do x:=x+1");
    test_can_all!(statement, while_leq, "while x<=10 do x:=x+1");

    test_can_all!(variable, simple, "x");
    test_can_all!(variable, array, "x[1]");

    test_can_all!(expression_list, single, "1");
    test_can_all!(expression_list, multiple_2, "1,2");
    test_can_all!(expression_list, multiple_3, "1,2,1+2");

    test_can_all!(expression, simple, "(+1)");
    test_can_all!(expression, relop_eq, "x = 2");
    test_can_all!(expression, relop_neq, "x <> 2");
    test_can_all!(expression, relop_le, "x < 2");
    test_can_all!(expression, relop_leq, "x <= 2");
    test_can_all!(expression, relop_gt, "x > 2");
    test_can_all!(expression, relop_gte, "x >= 2");
    // Composite from non-trivial simple exprs:
    // LHS: simple_expressino i mod 15, RHS: simple_expression 0
    test_can_all!(expression, relop_composite, "i mod 15 = 0");

    test_can_all!(simple_expression, term, "x");
    test_can_all!(simple_expression, sign_term, "-x");
    test_can_all!(simple_expression, add_op_1, "x + y");
    test_can_all!(simple_expression, add_op_2, "-x + y - z");

    test_can_all!(term, factor, "x");
    test_can_all!(term, mulop_single, "2*x");
    test_can_all!(term, mulop_multiple, "2*x*y");
    test_can_all!(term, mulop_single_star, "2*x");
    test_can_all!(term, mulop_single_slash, "2/x");
    test_can_all!(term, mulop_single_div, "4 DIV 2");
    test_can_all!(term, mulop_single_mod, "4 MOD 2");
    test_can_all!(term, mulop_single_mod_var_const, "i MOD 15");
    test_can_all!(term, mulop_single_and, "x AND y");

    test_can_all!(factor, id, "foo");
    test_can_all!(factor, id_list, "foo(x,y,z)");
    test_can_all!(factor, const_num, "42");
    test_can_all!(factor, const_character_string, "'foo'");
    test_can_all!(factor, parens, "(1)");
    test_can_all!(factor, not_id, "not x");
    test_can_all!(factor, not_id_list, "not foo(1,2)");
    test_can_all!(factor, not_num, "not 1");
    test_can_all!(factor, not_parens, "not (1+2)");
    test_can_all!(factor, not_not_parens, "not not (1+2)");

    test_can_all!(program, declarations_and_compound_statement_empty_program,
        "program helloWorld(output); var x:integer; begin end."
    );
    test_can_all!(program, subprogram_declarations_and_compound_statement_empty_program,
        r#"
        program helloWorld(output);
        function foo: integer; begin end;
        begin end."#
    );
    test_can_all!(program, compound_statement_empty_program, "program helloWorld(output);begin end.");
    test_can_all!(program, hello_world,
        r#"program helloWorld(output);begin writeLn('Hello, World!') end."#
    );

    #[test]
    fn parse_program_string_hello_world_returns_valid_il() {
        let actual = parse_program_string(r#"program helloWorld(output);begin writeLn('Hello, World!') end."#).unwrap();

        let expected = il::ProgramExpr::new(
            il::Id::new_from_str("helloWorld").unwrap(),
            il::IdentifierList::new(il::NonEmptyVec::single(il::Id::new_from_str("output").unwrap())),
            il::DeclarationsExpr::empty(),
            il::SubprogramDeclarations::empty(),
            il::CompoundStatement::new(vec![il::Statement::procedure(il::ProcedureStatement::with_params(
                il::Id::new_from_str("writeLn").unwrap(),
                il::ExpressionList::new(
                    il::NonEmptyVec::new(vec![il::Expression::simple(il::SimpleExpression::term(
                        il::Term::factor(il::Factor::string("Hello, World!")),
                    ))])
                        .unwrap(),
                ),
            ),
            )]),
        );

        assert_eq!(expected.id, actual.id);
        //assert_eq!(expected, actual);
    }

    #[test]
    fn parse_program_string_fizzbuzz_returns_valid_il() {
        let actual = parse_program_string(
            r#"
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
                          writeln(i)
                    end.
                "#).unwrap();

        // Just a smoke test for now (parsing not failing is already a good sign)
        assert_eq!(il::Id::new_from_str("fizzbuzz").unwrap(), actual.id);
        assert_eq!(DeclarationsExpr::new(vec![
            VarDeclaration::new(
                il::IdentifierList::new(
                    il::NonEmptyVec::single(il::Id::new_from_str("i").unwrap())),
                il::Type::standard(il::StandardType::Integer)),
        ]), actual.declarations);
    }
}
