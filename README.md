# Free Monad Interpreter for a Pascal Subset, in Rust

Writing a free monad interpreter for a tiny subset of Pascal, in Rust.
Let's see where this goes...

This project demonstrates elements of compiler and interpreter construction in Rust, including:

- using the **Pest** PEG (*programmable expression grammar*) parser generator to build a compiler front-end for a small subset
  of Pascal
- using the **free monad** pattern to build an interpreter for a simple language, in this case for pretty-printing Pascal
  programs


# Why Pascal

Pascal is a language from 1970 by Niklaus Wirth.
It is easy to build a compiler for Pascal, and has been used for teaching compilers extensively,
making grammars and related materials readily available, also for small but useful subsets of Pascal.

This makes it an ideal case, since it moves focus to the free monad interpreter and it applications
instead of consuming a lot of effort to design a bespoke language to use in the interpreter.

# Pretty Printer

The pretty printer is a "warm-up excercise". It is a function that takes a Pascal program AST and returns a string
representing the program source code, nicely formatted.

It consists of two parts: a recursive function that translates the AST to a simple printing language, a Free Monad, and
a Free Monad interpreter that interprets the printing language to a string to accomplish the formatting.

The translation from AST to printing language is done by a recursive function that is a good example of the continuation
passing style (CPS) that is also used in the Free Monad interpreter. Interestingly, CPS allows us to build data
structures such as lists and trees top down even though the data structures must be constructed bottom up.

It is defined in [`src/interpreters/pretty_printer.rs`](src/intepreters/pretty_printer.rs).

## Rust is not the Ideal Language for Free Monads
Lacking the `do` syntax or a `|>` operator in Rust, the composability of the Free Monad is not as nice as in Haskell or F#.
It could probably be improved with some `and_then` functions to chain the steps and align them vertically albeit 
a bit more verbosely.

Some people have tried to alleviate this using macros, see the list of literature below.

# Compiler Front-End

The compiler front-end takes a small subset of Pascal as input and produces an AST as output.
It is written with the [Pest](http://pest.rs) parser generator.

## Pest Grammar and PEG Parser

The grammar is specified in [`src/front_end/pascal_grammar.pest`](src/front_end/pascal_grammar.pest).

Normally Pascal grammars are given in EBNF with left-recursive rules (e.g. see the ANTLR grammar for Pascal and
the standards documents in the Literature section below). However, since Pest is a PEG parser generator,
we must adopt the grammar slightly to avoid left-recursion.

The grammar file is annotated to provide commentary on this.

That since Pest does not have a lexer/parser separation, we must provide special rules for *e.g.* whitespace.
For example, for the `WHITESPACE` rule, we mark it with `_` to indicate that it should match but not yield any tokens.
Then, for other rules we use the '@' atomic marker to indicate that they should not match whitespace inside them
in the few cases where this is needed.

### Ambiguities, Order of Rules and Look-Ahead

Identifiers and word-symbols, e.g. keywords such as `program` share the same space of strings, so we must use
negative lookahead to avoid matching keywords as identifiers and vice versa, see `IDENT` and `WORD_SYMBOL`.

Pest does not analyse the grammar for ambiguities, so we must do that ourselves. Here, a set of test cases
are handy, to ensure that we match productions correctly and that we consume the whole input for valid inputs.
In some cases, before elaborating the grammar, the parser would succeed but only consume part of the input.

For example, notice the use of `!COMMA` in the `expression_list` production below to ensure that we do not
match an `expression` only for an input with `expression COMMA expression`. So, it is not as eager in matching
as one might think.

```
expression_list = {
    expression ~ !COMMA
    | expression ~ (COMMA ~ expression_list)?
}
```

The order of the terms in a production matters.
For example, this works as expected, matching various relational operators in Pascal:

```
RELOP = {"=" | "<=" | ">=" | "<>" | "<" | ">" }
```

However, if we write it in this order, it does not work:

```
RELOP = {"=" | "<" | ">" | "<=" | ">=" | "<>"}
```

Notice the subtle difference: since `<` and `>` are prefixes of `<=` and `>=`,
but also individual tokens, the parser uses the first match, so in the latter case
it will match `<` and `>` for both `<`, `<=` and `>`and `>=`. It will not match `<=` and `>=`.

It is quirky and not very intuitive, but overall Pest is still quite nice to work with.

### Parser Code Should Take Ownership of the Parsing Result
It looks like Pest has been designed for the parsing code taking ownership of the result, 
rather than borrowing it.
The front-end code, however, was initially written to borrow the result. Because Pest 
parsing code uses the  `.into_inner()` method to get the result, which takes ownership,
we have to clone some Pairs sometimes. 

This can be avoided. See, *e.g.* the parsing code for `subprogram_declaration` that is written
using a style that fits better with Pest. Changing the parsing functions to take ownership
we could avoid cloning in the parsing stage.


# Literature

## Pascal

- Niklaus Wirth, "The Programming Language Pascal (Revised Report)". ETH Zürich,
    1973. https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/68910/eth-3059-01.pdf?sequence=1&isAllowed=y
- Charles Antony Richard Hoare; Niklaus Wirth, "An axiomatic definition of the programming language Pascal", ETH Zürich,
    1972. https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/68663/eth-3028-01.pdf?sequence=1&isAllowed=y
- Niklaus Wirth, "Pascal S: A Subset and its Implementation", ETH Zürich,
    1975. https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/68667/eth-3055-01.pdf
- The ANTLR parser generator has a grammar for Pascal: https://github.com/antlr/grammars-v4/tree/master/pascal
- A grammar for a small subset of [PASCAL](https://www2.seas.gwu.edu/~hchoi/teaching/cs160d/pascal.pdf)
- Pascal ISO 7185:1990, the unextended Pascal
  standard, https://github.com/antlr/grammars-v4/blob/master/pascal/iso7185.pdf
- Online Free Pascal compiler, https://www.onlinegdb.com/online_pascal_compiler

## Free Monads

- Introduction to Free Monads. Haskell. https://serokell.io/blog/introduction-to-free-monads
- CP Style Free Monads and Direct Style Free Monads.
  Idris. https://deque.blog/2017/12/08/continuation-passing-style-free-monads-and-direct-style-free-monads/
- Quentin Duval, Free Monads: from the basics to composable and effectful stream processing.
  Idris. https://quentinduval.github.io/blog/2017/11/13/free-monads-intro.html
- Functors in the Haskell Wiki, https://wiki.haskell.org/Category_theory/Functor
- Free structures (including Free Monads) in the Haskell Wiki, https://wiki.haskell.org/Free_structure
- Functors, Applicatives, And Monads In Pictures, https://www.adit.io/posts/2013-04-17-functors,_applicatives,_and_monads_in_pictures.html
- Scott Wlaschin,  Turtle Graphics in F# using Free Monads. https://fsharpforfunandprofit.com/posts/13-ways-of-looking-at-a-turtle-2/#way13
- Mark Seemann, Hello, Pure Command-Line Interaction, https://blog.ploeh.dk/2017/07/11/hello-pure-command-line-interaction/
- Mark Seemann, Combining Free Monads in Haskell, https://blog.ploeh.dk/2017/07/24/combining-free-monads-in-haskell/
- Higher Free macro for Rust, https://github.com/soulsource/higher-free-macro

## Notable Libraries

- The `pest` parser generator, used in the compiler front-end, see https://pest.rs/

# License

MIT License, see [`LICENSE`](LICENSE).

# Author

Martin Jul, 2023