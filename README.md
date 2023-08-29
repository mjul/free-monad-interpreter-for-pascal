# Free Monad Interpreter for a Pascal Subset, in Rust
Writing a free monad interpreter for a tiny subset of Pascal, in Rust.
Let's see where this goes...


# Why Pascal
Pascal is a language from 1970 by Niklaus Wirth. 
It is easy to build a compiler for Pascal, and has been used for teaching compilers extensively,
making grammars and related materials readily available, also for small but useful subsets of Pascal.

This makes it an ideal case, since it moves focus to the free monad interpreter and it applications
instead of consuming a lot of effort to design a bespoke language to use in the interpreter.

# Literature
## Pascal
- Niklaus Wirth, "The Programming Language Pascal (Revised Report)". ETH Zürich, 1973. https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/68910/eth-3059-01.pdf?sequence=1&isAllowed=y
- Charles Antony Richard Hoare; Niklaus Wirth, "An axiomatic definition of the programming language Pascal", ETH Zürich, 1972. https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/68663/eth-3028-01.pdf?sequence=1&isAllowed=y
- Niklaus Wirth, "Pascal S: A Subset and its Implementation", ETH Zürich, 1975. https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/68667/eth-3055-01.pdf
- The ANTLR parser generator has a grammar for Pascal: https://github.com/antlr/grammars-v4/tree/master/pascal
- A grammar for a small subset of [PASCAL](https://www2.seas.gwu.edu/~hchoi/teaching/cs160d/pascal.pdf):
## Free Monads
- Introduction to Free Monads. Haskell. https://serokell.io/blog/introduction-to-free-monads
- CP Style Free Monads and Direct Style Free Monads. Idris. https://deque.blog/2017/12/08/continuation-passing-style-free-monads-and-direct-style-free-monads/
- Quentin Duval, Free Monads: from the basics to composable and effectful stream processing. Idris. https://quentinduval.github.io/blog/2017/11/13/free-monads-intro.html
- Functors in the Haskell Wiki, https://wiki.haskell.org/Category_theory/Functor
- Free structures (including Free Monads) in the Haskell Wiki, https://wiki.haskell.org/Free_structure- Scott Wlaschin, Turtle Graphics in F# using Free Monads. https://fsharpforfunandprofit.com/posts/13-ways-of-looking-at-a-turtle-2/#way13 
- Mark Seemann, Hello, Pure Command-Line Interaction, https://blog.ploeh.dk/2017/07/11/hello-pure-command-line-interaction/
- Mark Seemann, Combining Free Monads in Haskell, https://blog.ploeh.dk/2017/07/24/combining-free-monads-in-haskell/
