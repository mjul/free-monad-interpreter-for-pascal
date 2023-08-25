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
- Niklaus Wirth, "The Programming Language Pascal (Revised Report)". ETH Zürich, 1973. https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/68910/eth-3059-01.pdf?sequence=1&isAllowed=y
- Charles Antony Richard Hoare; Niklaus Wirth, "An axiomatic definition of the programming language Pascal", ETH Zürich, 1972. https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/68663/eth-3028-01.pdf?sequence=1&isAllowed=y
- Niklaus Wirth, "Pascal S: A Subset and its Implementation", ETH Zürich, 1975. https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/68667/eth-3055-01.pdf
- A grammar for a small subset of [PASCAL](https://www2.seas.gwu.edu/~hchoi/teaching/cs160d/pascal.pdf):
