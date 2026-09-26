"""Correctness-gated benchmark for mojo-boolean-py.

Every case rebuilds the expected answer before anything is timed: the
byte-per-row NumPy evaluator, or the truth table produced by the real package's
own `expr(**kwargs)` where that is affordable.

One property of `boolean.py` shapes this file. Its expressions are DAGs with
heavy sharing: `a ^ b` built as `(a | b) & ~(a & b)` refers to `a` and `b`
twice, so a chain of `k` such gates has `2 ** k` root-to-leaf paths. Anything
that walks the tree recursively - including `Expression.symbols` - therefore
costs `2 ** k`, and a benchmark built from a long chain of shared gates spends
all its time in Python. The expressions here are built without sharing, the
program is compiled once outside the timing, and the baseline is given the same
expression.

Run with: python bench/bench.py
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import mojo_booleanpy as mb  # noqa: E402
from mojo_booleanpy import _lib  # noqa: E402


def _time(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def vector_evaluator(expr, columns):
    """Evaluate an expression tree column-wise, one uint8 array per symbol.

    This mirrors `boolean.py`'s own recursive `__call__`, with the recursion
    hoisted into NumPy. It is the fairest baseline for the bitsliced kernel: it
    is fully vectorised, it uses the same semantics, and the only thing it does
    not do is notice that eight rows fit in a byte.
    """
    import boolean

    from mojo_booleanpy.core import _is_false, _is_true

    def vec(node):
        if isinstance(node, boolean.Symbol):
            return columns[node.obj]
        if _is_true(node):
            return np.ones_like(next(iter(columns.values())))
        if _is_false(node):
            return np.zeros_like(next(iter(columns.values())))
        if isinstance(node, boolean.NOT):
            return (1 - vec(node.args[0])).astype(np.uint8)
        values = [vec(x) for x in node.args]
        out = values[0]
        for v in values[1:]:
            out = (out & v) if isinstance(node, boolean.AND) else (out | v)
        return out

    return vec(expr)


def standard_columns(names):
    n = 1 << len(names)
    rows = np.arange(n, dtype=np.uint64)
    return {
        name: ((rows >> (len(names) - 1 - i)) & 1).astype(np.uint8)
        for i, name in enumerate(names)
    }


def linear_expression(algebra, count, gates=1):
    """A share-free expression: every gate is a fresh node over the previous.

    `gates` layers of OR applied to disjoint symbol groups, folded from the
    right, gives an expression with `2 * gates * count - 1` nodes and no shared
    subtree, so walking it is linear.
    """
    symbols = algebra.symbols(*[f"s{i}" for i in range(count)])
    layer = list(symbols)
    for _ in range(gates):
        nxt = []
        for i in range(0, len(layer) - 1, 2):
            nxt.append(layer[i] | layer[i + 1])
        if len(layer) % 2:
            nxt.append(layer[-1])
        layer = nxt
    expr = layer[0]
    for extra in layer[1:]:
        expr = expr | extra
    return expr


def bench_truth_table(nsym: int = 14, gates: int = 3):
    """A 14-symbol, 3-layer expression over all 16384 rows of the table.

    The packed form does 256 word operations per operator where the byte form
    does 16384 byte operations.
    """
    import boolean

    a = boolean.BooleanAlgebra()
    expr = linear_expression(a, nsym, gates)
    program = mb.compile_expr(expr)
    names = program.names
    assert program.nsym == nsym
    columns = standard_columns(names)
    np.testing.assert_array_equal(
        mb.unpack(mb.run(program), program.nrows), vector_evaluator(expr, columns)
    )

    def baseline():
        return vector_evaluator(expr, columns)

    # The columns are hoisted out of the timing: they depend only on `nsym`, so
    # a caller evaluating many expressions of the same arity generates them
    # once. `bench_truth_columns` measures that step on its own.
    packed = _lib.truth_columns(nsym, program.nrows)

    def ours():
        return mb.unpack(mb.run(program, packed), program.nrows)

    return (
        f"truth_table n={nsym} 2^{nsym}",
        _time(baseline, 3),
        _time(ours, 3),
    )


def bench_truth_columns(nsym: int = 14):
    """Building the bit-packed columns, against NumPy's `packbits`.

    This is a once-per-arity setup cost rather than a per-evaluation one, and it
    is the one place the packed layout costs more than it saves.
    """
    nrows = 1 << nsym
    rows = np.arange(nrows, dtype=np.uint64)
    want = np.stack(
        [((rows >> (nsym - 1 - s)) & 1).astype(np.uint8) for s in range(nsym)], axis=1
    )

    def baseline():
        return np.packbits(want, axis=1, bitorder="little")

    got = _lib.truth_columns(nsym, nrows)
    np.testing.assert_array_equal(
        np.unpackbits(got.view(np.uint8), axis=1, bitorder="little")[:, :nrows].T, want
    )
    return (
        f"truth_columns n={nsym} 2^{nsym}",
        _time(baseline, 3),
        _time(lambda: _lib.truth_columns(nsym, nrows), 3),
    )


def bench_evaluate_rows(m: int = 1 << 20, nsym: int = 12):
    """A million rows and twelve symbols: 262144 words of input columns.

    Here the packing is pure overhead on the input side and pure win on the
    operator side, and the answer is wide enough that neither side fits in
    cache.
    """
    import boolean

    a = boolean.BooleanAlgebra()
    expr = linear_expression(a, nsym, gates=3)
    program = mb.compile_expr(expr)
    names = program.names
    rng = np.random.default_rng(1)
    rows = rng.integers(0, 2, size=(m, nsym)).astype(np.uint8)
    lookup = {name: rows[:, i] for i, name in enumerate(names)}
    np.testing.assert_array_equal(
        mb.evaluate_rows(expr, rows), vector_evaluator(expr, lookup)
    )

    def baseline():
        return vector_evaluator(expr, lookup)

    return (
        f"evaluate_rows m={m} n={nsym}",
        _time(baseline, 3),
        _time(lambda: mb.evaluate_rows(expr, rows), 3),
    )


def bench_minterms(nsym: int = 18, gates: int = 2):
    """Extracting the true rows out of a 2 ** 18 bit table."""
    import boolean

    a = boolean.BooleanAlgebra()
    expr = linear_expression(a, nsym, gates)
    program = mb.compile_expr(expr)
    names = program.names
    want = np.flatnonzero(vector_evaluator(expr, standard_columns(names)))
    words = mb.run(program)
    got = mb.minterms(expr)
    np.testing.assert_array_equal(got, want)
    assert _lib.popcount(words) == got.size

    def baseline():
        return np.flatnonzero(vector_evaluator(expr, standard_columns(names)))

    return (
        f"minterms n={nsym} 2^{nsym}",
        _time(baseline, 3),
        _time(lambda: mb.minterms(expr), 3),
    )


def main():
    print(f"{'case':<28}{'reference':>12}{'mojo-boolean-py':>18}{'ratio':>10}")
    print("-" * 70)
    for fn in (
        bench_truth_table,
        bench_truth_columns,
        bench_evaluate_rows,
        bench_minterms,
    ):
        label, ref, got = fn()
        ratio = ref / got if got else float("nan")
        print(f"{label:<28}{ref*1e3:>10.2f}ms{got*1e3:>16.2f}ms{ratio:>9.2f}x")


if __name__ == "__main__":
    main()
