"""Bitsliced truth tables and evaluation for `boolean.py` expressions.

`boolean.py` builds `Expression` trees out of `Symbol`, `NOT`, `AND` and `OR`,
and evaluates one assignment at a time through `expr(**kwargs)`. The functions
here take the same trees and evaluate all `2 ** n` assignments at once, 64
rows to a machine word, with the operators compiled once into a postfix stream.

    import boolean
    from mojo_booleanpy import truth_table, minterms

    algebra = boolean.BooleanAlgebra()
    a, b = algebra.symbols("a", "b")
    tt = truth_table(a & ~b)          # np.uint8, length 4
    minterms(a & ~b)                  # np.int64([1, 2])
"""

from __future__ import annotations

import functools
from collections import namedtuple

import numpy as np

from . import _lib

OP_NOT = _lib.OP_NOT
OP_AND = _lib.OP_AND
OP_OR = _lib.OP_OR
OP_TRUE = _lib.OP_TRUE
OP_FALSE = _lib.OP_FALSE

#: A compiled expression: the postfix stream, the symbol order, the stack the
#: stream needs, and the size of the table the standard columns produce.
#: Compiling is the part that walks the `boolean.py` tree, and walking that
#: tree is not cheap (see `symbol_names`), so a caller that evaluates the same
#: expression more than once should compile once and `run` the program.
Program = namedtuple("Program", "code names depth nsym nrows")


def _import_boolean():
    import boolean

    return boolean


def symbol_names(expr):
    """The symbols of an expression, in the order the truth table uses.

    The real package is the only source of the symbols; only the ordering and
    the traversal are this port's.

    `boolean.py` exposes its symbols as an unordered `set`
    (`Expression.symbols`), so a row index into a truth table would not be
    reproducible from it. This orders them by their printed name, which for the
    usual lower-case names is the order a reader would guess.

    The set is collected by walking the tree here rather than through
    `Expression.symbols`. That property reaches the answer via `get_literals`,
    which recurses over the whole tree and materialises a list of every literal
    it finds; on a chain of a few dozen operators that is thousands of
    duplicate nodes, and on a wider expression it dominates the cost of
    evaluating the expression at all. The set of `Symbol` nodes in the tree is
    the same set, and `test_symbol_names_match_the_real_package` checks that.
    """
    boolean = _import_boolean()
    seen: dict = {}
    stack = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, boolean.Symbol):
            seen.setdefault(node.obj, None)
        elif _is_true(node) or _is_false(node):
            continue
        else:
            stack.extend(node.args)
    return sorted(seen, key=str)


@functools.lru_cache(maxsize=1)
def _constant_types():
    """The `TRUE` / `FALSE` element classes, if this version still exposes them.

    `boolean.py` builds a fresh `TRUE` and `FALSE` for every `BooleanAlgebra`
    instance and stores them on the instance, so they cannot be compared by
    identity against a module-level constant; the classes behind them are the
    only stable handle. Looking them up defensively means a future rename turns
    into a clear `TypeError` on the unsupported node rather than a wrong
    truth table.
    """
    module = _import_boolean().boolean
    found = []
    for name in ("_TRUE", "_FALSE"):
        candidate = getattr(module, name, None)
        if isinstance(candidate, type):
            found.append(candidate)
    return tuple(found)


def _is_true(node) -> bool:
    types_ = _constant_types()
    return len(types_) == 2 and isinstance(node, types_[0])


def _is_false(node) -> bool:
    types_ = _constant_types()
    return len(types_) == 2 and isinstance(node, types_[1])


def compile_expr(expr) -> Program:
    """Compile an expression into the postfix stream the kernel consumes.

    A non-negative entry in `code` pushes the symbol with that index; the
    negatives are `OP_NOT`, `OP_AND`, `OP_OR` and the two constants `OP_TRUE` /
    `OP_FALSE`. `depth` is the maximum stack the postfix evaluation reaches.
    It is a named tuple, so `code, names, depth = compile_expr(e)` works.
    """
    boolean = _import_boolean()
    names = symbol_names(expr)
    index = {name: i for i, name in enumerate(names)}
    code: list[int] = []
    depth = 0
    peak = 0

    def walk(node):
        nonlocal depth, peak
        if isinstance(node, boolean.Symbol):
            code.append(index[node.obj])
            depth += 1
            peak = max(peak, depth)
            return
        if isinstance(node, boolean.NOT):
            for arg in node.args:
                walk(arg)
            code.append(OP_NOT)
            return
        if isinstance(node, (boolean.AND, boolean.OR)):
            # `boolean.py` lets AND and OR take any number of arguments, and
            # `DualBase.__call__` folds them left to right. The stream has to
            # say that explicitly: a single n-ary opcode would leave the kernel
            # with a stack depth that no longer matches the expression, and the
            # result would be read from a stale row.
            for arg in node.args:
                walk(arg)
            opcode = OP_AND if isinstance(node, boolean.AND) else OP_OR
            for _ in range(len(node.args) - 1):
                code.append(opcode)
                depth -= 1
            return
        if _is_true(node):
            code.append(OP_TRUE)
            depth += 1
            peak = max(peak, depth)
            return
        if _is_false(node):
            code.append(OP_FALSE)
            depth += 1
            peak = max(peak, depth)
            return
        raise TypeError(f"unsupported expression node: {type(node).__name__}")

    walk(expr)
    nsym = len(names)
    return Program(
        np.asarray(code, dtype=np.int32), names, peak, nsym, 1 << nsym
    )


def _as_int_array(assignments) -> np.ndarray:
    arr = np.asarray(assignments)
    if arr.ndim != 2:
        raise ValueError("assignments must be 2-D (m, nsym)")
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8)
    if arr.dtype.kind not in "iu":
        raise ValueError("assignments must be an integer or boolean array")
    if arr.size and (int(arr.max()) > 1 or int(arr.min()) < 0):
        # Two reductions, not `np.isin`: on a million rows `np.isin` against a
        # two-element set costs more than the whole rest of the call.
        raise ValueError("assignments must contain only 0 and 1")
    return arr.astype(np.uint8, copy=False)


def run(program: Program, columns: np.ndarray | None = None) -> np.ndarray:
    """Evaluate a compiled program over bit-packed columns, returning words.

    `columns` defaults to the standard truth-table columns for the program's
    own `nsym`. Pass explicit columns to evaluate over an arbitrary set of
    assignments; that is what `evaluate_rows` does.
    """
    if columns is None:
        columns = _lib.truth_columns(program.nsym, program.nrows)
    if program.depth == 0:
        return np.zeros(columns.shape[1], dtype=np.uint64)
    return _lib.eval_bitsliced(
        program.code, program.nsym, program.nrows, columns, program.depth
    )


def truth_table_words(expr) -> np.ndarray:
    """The packed truth table of an expression, as uint64 words.

    Bit `r` of the result is the value of the expression on assignment `r`.
    """
    program = compile_expr(expr)
    return run(program)


def truth_table(expr) -> np.ndarray:
    """The truth table of an expression as a uint8 array of length `2 ** n`.

    Row `r` assigns symbol `s` the value `(r >> (n - 1 - s)) & 1`.
    """
    program = compile_expr(expr)
    return unpack(run(program), program.nrows)


def unpack(words, nrows: int) -> np.ndarray:
    """Expand a packed table into one uint8 per row."""
    bits = np.unpackbits(np.asarray(words, dtype=np.uint64).view(np.uint8), bitorder="little")
    return bits[:nrows].astype(np.uint8)


def evaluate_rows(expr, assignments) -> np.ndarray:
    """Evaluate an expression on many assignments at once.

    `assignments` is an (m, n) array of 0/1 in the column order returned by
    `symbol_names`. The result is a uint8 array of length m. This is the same
    bitsliced kernel with a different set of input columns, so the operators
    still run 64 rows per word.
    """
    program = compile_expr(expr)
    names = program.names
    rows = _as_int_array(assignments)
    if rows.shape[1] != len(names):
        raise ValueError(
            f"assignments has {rows.shape[1]} columns but the expression uses "
            f"{len(names)} symbols"
        )
    m = rows.shape[0]
    if m == 0:
        return np.zeros(0, dtype=np.uint8)
    # Column s is the transposed, bit-packed version of column s of `rows`.
    # `packbits` with little-endian order puts bits 0..7 of a column in byte 0,
    # so the first eight bytes of a row are the first machine word, and it
    # zero-fills the tail of a column that is not a whole number of bytes.
    # `packbits` along a strided axis is four times slower than along a
    # contiguous one, so the transpose is made explicit rather than left to
    # `packbits` to walk backwards.
    packed = np.packbits(np.ascontiguousarray(rows.T), axis=1, bitorder="little")
    tail = packed.shape[1] % 8
    if tail:
        packed = np.pad(packed, ((0, 0), (0, 8 - tail)))
    columns = np.ascontiguousarray(packed).view(np.uint64)
    if program.depth == 0:
        return np.zeros(m, dtype=np.uint8)
    words = _lib.eval_bitsliced(program.code, len(names), m, columns, program.depth)
    return np.unpackbits(words.view(np.uint8), bitorder="little")[:m].astype(np.uint8)


def _mask_tail(words: np.ndarray, nrows: int) -> np.ndarray:
    """Clear the bits above row `nrows - 1` in a packed table.

    The evaluator already does this on the way out; the complement in
    `maxterms` needs it done again.
    """
    tail = nrows & 63
    if tail and words.size:
        words = words.copy()
        words[-1] &= np.uint64((1 << tail) - 1)
    return words


def minterms(expr) -> np.ndarray:
    """Row indices on which the expression is true, ascending.

    These are the minterms of the function, read straight out of the packed
    truth table.
    """
    return _lib.true_indices(truth_table_words(expr))


def maxterms(expr) -> np.ndarray:
    """Row indices on which the expression is false, ascending.

    Complementing a packed table inverts the padding bits above row
    `2 ** n - 1` too, so the tail has to be cleared again before scanning;
    otherwise every position past the end of the table comes back as a
    maxterm.
    """
    names = symbol_names(expr)
    words = ~truth_table_words(expr)
    nrows = 1 << len(names)
    return _lib.true_indices(_mask_tail(words, nrows))


def count_true(expr) -> int:
    """How many of the `2 ** n` assignments satisfy the expression."""
    return _lib.popcount(truth_table_words(expr))
