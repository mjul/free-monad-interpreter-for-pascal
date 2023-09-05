//! Front end for the Pascal interpreter.
//! This module parses Pascal files to the intermediate representation.
//!
//! It uses the Pest parser generator to parse the Pascal files,
//! see <https://pest.rs/>

use pest::iterators::Pair;
use pest::Parser;
use pest_derive::Parser;

use crate::il;

#[derive(Parser)]
#[grammar = "src/front_end/pascal_grammar.pest"]
pub struct PascalParser;

#[derive(Debug)]
enum FrontEndError {
    ParseError(pest::error::Error<Rule>),
    ConversionError(ConversionError),
}

#[derive(Debug)]
enum ConversionError {
    ParseResultNotSingular,
    UnexpectedRuleInPair(Rule),
    ConversionError(String),
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
        Rule::declarations => match pair.clone().into_inner().next() {
            None => Ok(il::DeclarationsExpr::empty()),
            Some(p) => todo!(),
        },
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
                    _ => todo!("il_statement_from inner: {:?}", first_inner.as_rule()),
                },
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
            match inners.len() {
                0 => Err(ConversionError::ConversionError(
                    "Unexpected empty expression".to_string(),
                )),
                1 => {
                    let se = il_simple_expression_from(&inners[0])?;
                    Ok(il::Expression::simple(se))
                }
                _ => todo!("il_expression_from: {:?}", inners),
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
            match inners.len() {
                0 => Err(ConversionError::ConversionError(
                    "Unexpected empty term".to_string(),
                )),
                1 => {
                    let f = il_factor_from(&inners[0])?;
                    Ok(il::Term::factor(f))
                }
                _ => todo!("il_term_from: {:?}", inners),
            }
        }
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
        Rule::sign => match pair.as_str() {
            "+" => Ok(il::Sign::Plus),
            "-" => Ok(il::Sign::Minus),
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

fn parse_program_string(input: &str) -> Result<il::ProgramExpr, FrontEndError> {
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
        ($name:ident, $rule:ident, $input:expr) => {
            paste!{
                #[test]
                fn [<pascal_parser_can_parse_ $name _without_err>]() {
                    let result_pairs = PascalParser::parse(Rule::$rule, $input);
                    dbg!(&result_pairs);
                    assert!(result_pairs.is_ok());
                    assert_parse_consumed_all_input(result_pairs.unwrap(), &Rule::$rule, $input);
                }
            }
        };
    }

    test_can_all!(ident_mixed_case, IDENT, "helloWorld");

    test_can_all!(string_literal_empty, STRING_LITERAL, "''");
    test_can_all!(string_literal_simple, STRING_LITERAL, "'abc'");
    test_can_all!(string_literal_with_quoted_quote, STRING_LITERAL, "'foo''bar'");

    test_can_all!(keyword_program, PROGRAM, "program");
    test_can_all!(keyword_program_mixed_case, PROGRAM, "pRoGrAm");

    test_can_all!(identifier_list_single_id, identifier_list, "output");
    test_can_all!(identifier_list_multiple_ids, identifier_list, "a,b,c");

    test_can_all!(declarations_empty, declarations, "");
    test_can_all!(declarations_single_decl_single_var, declarations, "var a : integer;");
    test_can_all!(declarations_single_dec_multiple_vars, declarations, "var i, j : integer;");
    test_can_all!(declarations_multiple_decls, declarations, "var a : integer; var b: integer;");

    test_can_all!(subprogram_declarations_empty, subprogram_declarations, "");
    test_can_all!(subprogram_declarations_single_function, subprogram_declarations, "function foo: integer; begin end;");

    test_can_all!(subprogram_head_function_no_args, subprogram_head, "function foo: integer;");
    test_can_all!(subprogram_head_function_with_args, subprogram_head, "function foo(x:integer): integer;");
    test_can_all!(subprogram_head_procedure_no_args, subprogram_head, "procedure foo;");
    test_can_all!(subprogram_head_procedure_with_args, subprogram_head, "procedure foo(x:integer);");

    test_can_all!(arguments_empty, arguments, "");
    test_can_all!(arguments_single, arguments, "(x:integer)");
    test_can_all!(arguments_multiple, arguments, "(x:integer; y : integer)");

    test_can_all!(parameter_list_single, parameter_list, "x:integer");
    test_can_all!(parameter_list_multiple_2, parameter_list, "x:integer; y : integer");
    test_can_all!(parameter_list_multiple_3, parameter_list, "x:integer; y : integer; z : integer");

    test_can_all!(begin, BEGIN, "begin");
    test_can_all!(end, END, "end");

    test_can_all!(optional_statements_empty, optional_statements, "");
    test_can_all!(optional_statements_single_statement, optional_statements, "x := 1");
    test_can_all!(optional_statements_multiple_statements, optional_statements, "x:=1; y:=2");

    test_can_all!(compound_statement_empty, compound_statement, "begin end");
    test_can_all!(compound_statement_single_assignment, compound_statement, "begin x:=1 end");
    test_can_all!(compound_statement_single_writeln, compound_statement, "begin writeLn('Hello, World!') end");
    test_can_all!(compound_statement_multiple, compound_statement, "begin x:=1; if x>10 then x:=10 else x:=x end");

    test_can_all!(statement_assignment, statement, "x:=1");
    test_can_all!(statement_procedure_statement, statement, "writeLn('Hello, World!')");
    test_can_all!(statement_compound_statement, statement, "begin x:=1;y:=2 end");
    test_can_all!(statement_if_then_else, statement, "if x>10 then x:=10 else x:=x");
    test_can_all!(statement_while, statement, "while x<10 do x:=x+1");
    test_can_all!(statement_while_leq, statement, "while x<=10 do x:=x+1");

    test_can_all!(variable_simple, variable, "x");
    test_can_all!(variable_array, variable, "x[1]");

    test_can_all!(expression_list_single, expression_list, "1");
    test_can_all!(expression_list_multiple_2, expression_list, "1,2");
    test_can_all!(expression_list_multiple_3, expression_list, "1,2,1+2");

    test_can_all!(expression_simple, expression, "(+1)");
    test_can_all!(expression_relop_eq, expression, "x = 2");
    test_can_all!(expression_relop_neq, expression, "x <> 2");
    test_can_all!(expression_relop_le, expression, "x < 2");
    test_can_all!(expression_relop_leq, expression, "x <= 2");
    test_can_all!(expression_relop_gt, expression, "x > 2");
    test_can_all!(expression_relop_gte, expression, "x >= 2");

    test_can_all!(simple_expression_term, simple_expression, "x");
    test_can_all!(simple_expression_sign_term, simple_expression, "-x");
    test_can_all!(simple_expression_add_op_1, simple_expression, "x + y");
    test_can_all!(simple_expression_add_op_2, simple_expression, "-x + y - z");

    test_can_all!(term_factor, term, "x");
    test_can_all!(term_mulop_single, term, "2*x");
    test_can_all!(term_mulop_multiple, term, "2*x*y");

    test_can_all!(factor_id, factor, "foo");
    test_can_all!(factor_id_list, factor, "foo(x,y,z)");
    test_can_all!(factor_const_num, factor, "42");
    test_can_all!(factor_const_character_string, factor, "'foo'");
    test_can_all!(factor_parens, factor, "(1)");
    test_can_all!(factor_not_id, factor, "not x");
    test_can_all!(factor_not_id_list, factor, "not foo(1,2)");
    test_can_all!(factor_not_num, factor, "not 1");
    test_can_all!(factor_not_parens, factor, "not (1+2)");
    test_can_all!(factor_not_not_parens, factor, "not not (1+2)");

    test_can_all!(program_declarations_and_compound_statement_empty_program, program, "program helloWorld(output); var x:integer; begin end.");
    test_can_all!(program_subprogram_declarations_and_compound_statement_empty_program, program,
        r#"
        program helloWorld(output);
        function foo: integer; begin end;
        begin end."#
    );
    test_can_all!(program_compound_statement_empty_program, program, "program helloWorld(output);begin end.");
    test_can_all!(program_hello_world, program, r#"program helloWorld(output);begin writeLn('Hello, World!') end."#);


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
}
