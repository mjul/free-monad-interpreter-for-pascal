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
use crate::il::{DeclarationsExpr, ParameterGroup, StandardType, SubprogramHead, Type};

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

impl std::error::Error for FrontEndError {}

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

impl std::error::Error for ConversionError {}

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
    // Try another matching strategy, implementing it by peeking rather than cloning and iterating
    match &pair.as_rule() {
        Rule::subprogram_declarations => {
            dbg!(&pair.as_str(), &pair);
            let mut inners = pair.clone().into_inner().into_iter().peekable();
            match inners.peek() {
                None => Ok(il::SubprogramDeclarations::empty()),
                Some(_p) => {
                    let mut subprograms = vec![];
                    while let Some(p) = inners.next() {
                        match p.as_rule() {
                            Rule::subprogram_declaration => {
                                let sd = il_subprogram_declaration_from(&p)?;
                                subprograms.push(sd);
                            }
                            Rule::SEMICOLON => { /* ignore terminators */ }
                            _ => {
                                return Err(ConversionError::UnexpectedRuleInPair(
                                    p.as_rule(),
                                ));
                            }
                        }
                    }
                    Ok(il::SubprogramDeclarations::new(subprograms))
                }
            }
        }
        _ => Err(ConversionError::UnexpectedRuleInPair(pair.as_rule())),
    }
}

fn il_subprogram_declaration_from(pair: &Pair<Rule>) -> Result<il::SubprogramDeclaration, ConversionError> {
    // Another non-cloning strategy, using .next() without peeking
    // We still have to clone to be consistent with the other functions that all borrow the Pair.
    // If we refactor to taking ownership of the Pair, we can avoid cloning, but the we should do it for
    // all the parsing functions.
    match &pair.as_rule() {
        Rule::subprogram_declaration => {
            let mut inners = pair.clone().into_inner();
            match (inners.next(), inners.next(), inners.next(), inners.next()) {
                (Some(subprogram_head), Some(declarations), Some(compound_statement), None) => {
                    let sh = il_subprogram_head_from(&subprogram_head)?;
                    let ds = il_declarations_from(&declarations)?;
                    let cs = il_compound_statement_from(&compound_statement)?;
                    Ok(il::SubprogramDeclaration::new(sh, ds, cs))
                }
                _ => Err(ConversionError::ConversionError("Unexpected inner pairs under subprogram_declaration rule".to_string())),
            }
        }
        _ => Err(ConversionError::UnexpectedRuleInPair(pair.as_rule())),
    }
}

fn il_subprogram_head_from(pair: &Pair<Rule>) -> Result<SubprogramHead, ConversionError> {
    match &pair.as_rule() {
        Rule::subprogram_head => {
            // If we take ownership of the pair instead of borrowing it, we can avoid cloning
            let mut inners = pair.clone().into_inner();
            match (inners.next(), inners.next(), inners.next(), inners.next(), inners.next(), inners.next(), inners.next()) {
                (Some(_function), Some(id), Some(arguments), Some(_colon), Some(standard_type), Some(_semicolon), None) => {
                    let id = il_id_from(&id)?;
                    let params = il_parameter_groups_from_arguments(&arguments)?;
                    let ty = il_standard_type_from(&standard_type)?;
                    Ok(SubprogramHead::function(id, params, ty))
                }
                (Some(_procedure), Some(id), Some(arguments), Some(_semicolon), None, None, None) => {
                    let id = il_id_from(&id)?;
                    let params = il_parameter_groups_from_arguments(&arguments)?;
                    Ok(SubprogramHead::procedure(id, params))
                }
                _ => Err(ConversionError::ConversionError("Unexpected inner pairs under subprogram_head rule".to_string())),
            }
        }
        _ => Err(ConversionError::UnexpectedRuleInPair(pair.as_rule())),
    }
}

/// This reads the arguments into a possibly empty vector of [ParameterGroup]s.
fn il_parameter_groups_from_arguments(pair: &Pair<Rule>) -> Result<Vec<ParameterGroup>, ConversionError> {
    match &pair.as_rule() {
        Rule::arguments => {
            // The production: arguments -> ( parameter_list ) | epsilon
            // We have to clone since the pair is borrowed, not owned
            let mut inners = pair.clone().into_inner().peekable();
            match inners.peek().map(|p| p.as_rule()) {
                None => Ok(vec![]),
                Some(Rule::LPAREN) => {
                    // Production variant: arguments -> ( parameter_list )
                    inners.next(); // consume the LPAREN
                    match (inners.next(), inners.next()) {
                        (Some(pl), Some(_rparen)) => {
                            let params = il_parameter_groups_from_parameter_list(&pl)?;
                            Ok(params)
                        }
                        _ => Err(ConversionError::ConversionError(format!("Unexpected inner pairs under arguments rule: {:?}", pair))),
                    }
                }
                _ => Err(ConversionError::ConversionError(format!("Unexpected inner pairs under arguments rule: {:?}", pair))),
            }
        }
        _ => Err(ConversionError::UnexpectedRuleInPair(pair.as_rule())),
    }
}

/// This reads the arguments into a possibly empty vector of [ParameterGroup]s.
fn il_parameter_groups_from_parameter_list(pair: &Pair<Rule>) -> Result<Vec<ParameterGroup>, ConversionError> {
    match &pair.as_rule() {
        // Cloning is necessary since we have only borrowed the Pair
        Rule::parameter_list => {
            let mut inners = pair
                .clone()
                .into_inner()
                .into_iter()
                .peekable();
            let mut result = vec![];
            while let Some(p) = inners.next() {
                todo!("il_parameter_groups_from_parameter_list: {:?}", p.as_rule());
            }
            Ok(result)
        }
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
                    let _t = il_term_from(&next2)?;
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
    dbg!(&pair.as_str(), &pair);
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
                        Rule::unsigned_constant => {
                            let inners = p.clone().into_inner().collect::<Vec<Pair<Rule>>>();
                            match &inners[..] {
                                [p] => match p.as_rule() {
                                    Rule::unsigned_number => {
                                        let n = p.as_str().parse::<i32>().map_err(|e| {
                                            ConversionError::ConversionError(format!(
                                                "Failed to parse unsigned_number: {}",
                                                e
                                            ))
                                        })?;
                                        Ok(il::Factor::number(n))
                                    }
                                    Rule::character_string => {
                                        let inners = p.clone().into_inner().collect::<Vec<Pair<Rule>>>();
                                        match &inners[..] {
                                            [p] => match p.as_rule() {
                                                Rule::STRING_LITERAL => {
                                                    let s = string_from_string_literal(p)?;
                                                    Ok(il::Factor::string(s))
                                                }
                                                _ => Err(ConversionError::UnexpectedRuleInPair(p.as_rule())),
                                            },
                                            _ => unimplemented!("il_factor_from: character_string: {:?}", inners),
                                        }
                                    }
                                    _ => Err(ConversionError::UnexpectedRuleInPair(p.as_rule())),
                                },
                                _ => todo!("il_factor_from: unsigned_constant from: {:?}", p),
                            }
                        }
                        _ => Err(ConversionError::UnexpectedRuleInPair(p.as_rule()))
                    }
                }
                [lparen, expr, rparen] => {
                    match (lparen.as_rule(), expr.as_rule(), rparen.as_rule()) {
                        (Rule::LPAREN, Rule::expression, Rule::RPAREN) => {
                            let e = il_expression_from(expr)?;
                            Ok(il::Factor::parens(e))
                        }
                        _ => Err(ConversionError::UnexpectedRuleInPair(lparen.as_rule()))
                    }
                }
                [ident, lparen, expr_list, rparen] => {
                    match (ident.as_rule(), lparen.as_rule(), expr_list.as_rule(), rparen.as_rule()) {
                        (Rule::IDENT, Rule::LPAREN, Rule::expression_list, Rule::RPAREN) => {
                            let id = il_id_from(ident)?;
                            let el = il_expression_list_from(expr_list)?;
                            Ok(il::Factor::id_with_params(id, el))
                        }
                        _ => Err(ConversionError::UnexpectedRuleInPair(lparen.as_rule()))
                    }
                }
                // TODO: not factor is missing (two inners)
                _ => todo!("il_factor_from: inners {:?}", inners),
            }
        }
        _ => Err(ConversionError::UnexpectedRuleInPair(pair.as_rule())),
    }
}

/// Convert a Pascal string literal to a Rust string.
fn string_from_string_literal(pair: &Pair<Rule>) -> Result<String, ConversionError> {
    match pair.as_rule() {
        Rule::STRING_LITERAL => {
            let mut result = String::new();
            let mut cursor = pair.as_str().chars().into_iter().peekable();
            while let Some(ch) = cursor.next() {
                let next_ch = cursor.peek();
                match (ch, next_ch) {
                    ('\'', Some('\'')) => {
                        result.push('\'');
                        // consume the second quote
                        cursor.next();
                    }
                    ('\'', _) => { /* start or end quote, ignore and advance to next */ }
                    (c, _) => result.push(c.clone()),
                }
            }
            Ok(result)
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

    use crate::il::{CompoundStatement, DeclarationsExpr, ExpressionList, Id, NonEmptyVec, VarDeclaration};
    use crate::il::Statement::Assignment;

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
    test_can_all!(parameter_list, single_with_multiple_ids, "x,y,z : integer");
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
    test_can_all!(compound_statement, single_fib, "begin if (n <= 2) then fib:=1 else fib:=fib(n-1)+fib(n-2) end");
    test_can_all!(compound_statement, multiple, "begin x:=1; if x>10 then x:=10 else x:=x end");

    test_can_all!(statement, assignment_constant, "x:=1");
    test_can_all!(statement, assignment_function_call, "x:=fib(n-1)");
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
    test_can_all!(factor, id_list_fib, "fib(n-1)");
    test_can_all!(factor, const_num, "42");
    test_can_all!(factor, const_character_string, "'foo'");
    test_can_all!(factor, parens_of_const, "(1)");
    test_can_all!(factor, parens_of_expr, "(1+2)");
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
    fn string_from_string_literal_with_no_escaped_quotes_returns_inner_string() {
        let parsed = PascalParser::parse(Rule::STRING_LITERAL, "'foo'").unwrap();
        let pair = parsed.into_iter().next().unwrap();
        let actual = string_from_string_literal(&pair).unwrap();
        assert_eq!("foo", actual);
    }

    #[test]
    fn string_from_string_literal_with_escaped_quotes_returns_unescaped_inner_string() {
        let parsed = PascalParser::parse(Rule::STRING_LITERAL, "'foo''bar''''baz'").unwrap();
        let pair = parsed.into_iter().next().unwrap();
        let actual = string_from_string_literal(&pair).unwrap();
        assert_eq!("foo'bar''baz", actual);
    }

    #[test]
    fn il_factor_from_from_pascal_parser_parse_factor_of_const_string_has_right_type() {
        let parsed = PascalParser::parse(Rule::factor, "'Hello, World!'").unwrap();
        let actual = il_factor_from(&parsed.into_iter().next().unwrap()).unwrap();
        assert_eq!(il::Factor::string("Hello, World!".to_string()), actual);
    }

    #[test]
    fn il_factor_from_from_pascal_parser_parse_factor_of_const_string_with_escapes_has_right_type() {
        let parsed = PascalParser::parse(Rule::factor, "'abc''def'").unwrap();
        let actual = il_factor_from(&parsed.into_iter().next().unwrap()).unwrap();
        assert_eq!(il::Factor::string("abc'def".to_string()), actual);
    }

    #[test]
    fn il_factor_from_from_pascal_parser_parse_factor_of_const_number_has_right_type() {
        let parsed = PascalParser::parse(Rule::factor, "42").unwrap();
        dbg!(&parsed);
        let actual = il_factor_from(&parsed.into_iter().next().unwrap()).unwrap();
        assert_eq!(il::Factor::number(42), actual);
    }

    #[test]
    fn il_factor_from_from_pascal_parser_parse_factor_of_identifier_has_right_type() {
        let parsed = PascalParser::parse(Rule::factor, "x").unwrap();
        let actual = il_factor_from(&parsed.into_iter().next().unwrap()).unwrap();
        assert_eq!(il::Factor::id(Id::new_from_str("x").unwrap()), actual);
    }

    #[test]
    fn il_factor_from_from_pascal_parser_parse_factor_of_function_call_has_right_type() {
        let parsed = PascalParser::parse(Rule::factor, "foo(x)").unwrap();
        let actual = il_factor_from(&parsed.into_iter().next().unwrap()).unwrap();

        assert_eq!(
            il::Factor::id_with_params(
                Id::new_from_str("foo").unwrap(),
                ExpressionList::new(
                    NonEmptyVec::single(
                        il::Expression::simple(
                            il::SimpleExpression::term(
                                il::Term::factor(
                                    il::Factor::id(il::Id::new_from_str("x").unwrap()))))))),
            actual);
    }

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
                        il::Term::factor(il::Factor::string("Hello, World!".to_string())),
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

        let CompoundStatement(stmts) = actual.compound_statement;

        assert_eq!(Assignment(il::AssignmentStatement::new(
            il::Variable::id(il::Id::new_from_str("i").unwrap()),
            il::Expression::simple(il::SimpleExpression::term(il::Term::factor(il::Factor::number(1)))))),
                   stmts[0]);
    }
}
