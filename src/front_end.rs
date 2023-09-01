//! Front end for the Pascal interpreter.
//! This module parses Pascal files to the intermediate representation.
//!
//! It uses the Pest parser generator to parse the Pascal files,
//! see https://pest.rs/

use pest::Parser;
use pest_derive::Parser;

#[derive(Parser)]
#[grammar = "src/front_end/pascal_grammar.pest"]
pub struct PascalParser;


#[cfg(test)]
mod tests {
    use pest::iterators::Pairs;
    use super::*;

    #[test]
    fn pascal_parser_can_parse_whitespace_multiple_without_err() {
        let result_pairs = PascalParser::parse(Rule::WHITESPACE, "  \t\r\n");
        assert!(result_pairs.is_ok());
        assert_eq!(0, result_pairs.unwrap().count(), "Expected whitespace to parse to no tokens");
    }

    fn assert_parse_consumed_all_input(pairs: Pairs<Rule>, rule: &Rule, input: &str) {
        assert_eq!(1, pairs.len());
        let p = pairs.into_iter().next().unwrap();
        assert_eq!(rule, &p.as_rule(), "Expected to parse rule {:?}", rule);
        assert_eq!(input, p.as_span().as_str(), "Expected to consume all input");
    }

    macro_rules! test_can_all {
        ($name:ident, $rule:ident, $input:expr) => {
            #[test]
            fn $name() {
                let result_pairs = PascalParser::parse(Rule::$rule, $input);
                dbg!(&result_pairs);
                assert!(result_pairs.is_ok());
                assert_parse_consumed_all_input(result_pairs.unwrap(), &Rule::$rule, $input);
            }
        };
    }

    test_can_all!(pascal_parser_can_parse_ident_mixed_case_without_err, IDENT, "helloWorld");

    test_can_all!(pascal_parser_can_parse_string_literal_empty_without_err, STRING_LITERAL, "''");
    test_can_all!(pascal_parser_can_parse_string_literal_simple_without_err, STRING_LITERAL, "'abc'");
    test_can_all!(pascal_parser_can_parse_string_literal_with_quoted_quote_without_err, STRING_LITERAL, "'foo''bar'");

    test_can_all!(pascal_parser_can_parse_keyword_program_without_err, PROGRAM, "program");
    test_can_all!(pascal_parser_can_parse_keyword_program_mixed_case_without_err, PROGRAM, "pRoGrAm");

    test_can_all!(pascal_parser_can_parse_identifier_list_single_id_without_err, identifier_list, "output");
    test_can_all!(pascal_parser_can_parse_identifier_list_multiple_ids_without_err, identifier_list, "a,b,c");

    test_can_all!(pascal_parser_can_parse_declarations_empty_without_err, declarations, "");
    test_can_all!(pascal_parser_can_parse_declarations_single_decl_single_var_without_err, declarations, "var a : integer;");
    test_can_all!(pascal_parser_can_parse_declarations_single_dec_multiple_vars_without_err, declarations, "var i, j : integer;");
    test_can_all!(pascal_parser_can_parse_declarations_multiple_decls_without_err, declarations, "var a : integer; var b: integer;");

    test_can_all!(pascal_parser_can_parse_subprogram_declarations_empty_without_err, subprogram_declarations, "");
    test_can_all!(pascal_parser_can_parse_subprogram_declarations_single_function_without_err, subprogram_declarations, "function foo: integer; begin end;");

    test_can_all!(pascal_parser_can_parse_subprogram_head_function_no_args_without_err, subprogram_head, "function foo: integer;");
    test_can_all!(pascal_parser_can_parse_subprogram_head_function_with_args_without_err, subprogram_head, "function foo(x:integer): integer;");
    test_can_all!(pascal_parser_can_parse_subprogram_head_procedure_no_args_without_err, subprogram_head, "procedure foo;");
    test_can_all!(pascal_parser_can_parse_subprogram_head_procedure_with_args_without_err, subprogram_head, "procedure foo(x:integer);");

    test_can_all!(pascal_parser_can_parse_arguments_empty_without_err, arguments, "");
    test_can_all!(pascal_parser_can_parse_arguments_single_without_err, arguments, "(x:integer)");
    test_can_all!(pascal_parser_can_parse_arguments_multiple_without_err, arguments, "(x:integer; y : integer)");

    test_can_all!(pascal_parser_can_parse_parameter_list_single_without_err, parameter_list, "x:integer");
    test_can_all!(pascal_parser_can_parse_parameter_list_multiple_2_without_err, parameter_list, "x:integer; y : integer");
    test_can_all!(pascal_parser_can_parse_parameter_list_multiple_3_without_err, parameter_list, "x:integer; y : integer; z : integer");

    test_can_all!(pascal_parser_can_parse_begin_without_err, BEGIN, "begin");
    test_can_all!(pascal_parser_can_parse_end_without_err, END, "end");

    test_can_all!(pascal_parser_can_parse_optional_statements_empty_without_err, optional_statements, "");
    test_can_all!(pascal_parser_can_parse_optional_statements_single_statement_without_err, optional_statements, "x := 1");
    test_can_all!(pascal_parser_can_parse_optional_statements_multiple_statements_without_err, optional_statements, "x:=1; y:=2");

    test_can_all!(pascal_parser_can_parse_compound_statement_empty_without_err, compound_statement, "begin end");
    test_can_all!(pascal_parser_can_parse_compound_statement_single_assignment_without_err, compound_statement, "begin x:=1 end");
    test_can_all!(pascal_parser_can_parse_compound_statement_single_writeln_without_err, compound_statement, "begin writeLn('Hello, World!') end");
    test_can_all!(pascal_parser_can_parse_compound_statement_multiple_without_err, compound_statement, "begin x:=1; if x>10 then x:=10 else x:=x end");

    test_can_all!(pascal_parser_can_parse_statement_assignment, statement, "x:=1");
    test_can_all!(pascal_parser_can_parse_statement_procedure_statement, statement, "writeLn('Hello, World!')");
    test_can_all!(pascal_parser_can_parse_statement_compound_statement, statement, "begin x:=1;y:=2 end");
    test_can_all!(pascal_parser_can_parse_statement_if_then_else, statement, "if x>10 then x:=10 else x:=x");
    test_can_all!(pascal_parser_can_parse_statement_while, statement, "while x<10 do x:=x+1");

    test_can_all!(pascal_parser_can_parse_variable_simple_without_err, variable, "x");
    test_can_all!(pascal_parser_can_parse_variable_array_without_err, variable, "x[1]");

    test_can_all!(pascal_parser_can_parse_expression_list_single_without_err, expression_list, "1");
    test_can_all!(pascal_parser_can_parse_expression_list_multiple_2_without_err, expression_list, "1,2");
    test_can_all!(pascal_parser_can_parse_expression_list_multiple_3_without_err, expression_list, "1,2,1+2");

    test_can_all!(pascal_parser_can_parse_expression_simple_without_err, expression, "(+1)");
    test_can_all!(pascal_parser_can_parse_expression_relop_without_err, expression, "x < 2");

    test_can_all!(pascal_parser_can_parse_simple_expression_term_without_err, simple_expression, "x");
    test_can_all!(pascal_parser_can_parse_simple_expression_sign_term_without_err, simple_expression, "-x");
    test_can_all!(pascal_parser_can_parse_simple_expression_add_op_1_without_err, simple_expression, "x + y");
    test_can_all!(pascal_parser_can_parse_simple_expression_add_op_2_without_err, simple_expression, "-x + y - z");

    test_can_all!(pascal_parser_can_parse_term_factor_without_err, term, "x");
    test_can_all!(pascal_parser_can_parse_term_mulop_single_without_err, term, "2*x");
    test_can_all!(pascal_parser_can_parse_term_mulop_multiple_without_err, term, "2*x*y");

    test_can_all!(pascal_parser_can_parse_factor_id_without_err, factor, "foo");
    test_can_all!(pascal_parser_can_parse_factor_id_list_without_err, factor, "foo(x,y,z)");
    test_can_all!(pascal_parser_can_parse_factor_num_without_err, factor, "42");
    test_can_all!(pascal_parser_can_parse_factor_parens_without_err, factor, "(1)");
    test_can_all!(pascal_parser_can_parse_factor_not_without_err, factor, "not x");

    test_can_all!(pascal_parser_can_parse_program_empty_program_without_err, program, "program helloWorld(output);.");
    test_can_all!(pascal_parser_can_parse_program_hello_world_without_err, program, r#"program helloWorld(output);begin writeLn('Hello, World!') end."#);
}
