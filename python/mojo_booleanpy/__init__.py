"""mojo-boolean-py: bitsliced truth tables for `boolean.py` expressions.

Installable alongside the real `boolean` package, which it is tested against
for parity. The tree rewriting, the parser and the DNF/CNF normalisation stay
in `boolean`; only evaluation is ported.
"""

from .core import (
    Program,
    Program,
    compile_expr,
    count_true,
    evaluate_rows,
    maxterms,
    minterms,
    run,
    unpack,
    symbol_names,
    truth_table,
    truth_table_words,
)

__all__ = [
    "Program",
    "compile_expr",
    "count_true",
    "evaluate_rows",
    "maxterms",
    "minterms",
    "run",
    "symbol_names",
    "truth_table",
    "truth_table_words",
    "unpack",
]
__version__ = "0.1.0"
