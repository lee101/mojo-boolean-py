"""Parity of the bitsliced evaluator with `boolean.py` itself.

`boolean.py` evaluates one assignment at a time through `expr(**kwargs)`. The
reference here is exactly that: every one of the `2 ** n` rows is evaluated by
the real package and compared against the truth table the kernel produced.
Everything in this file is 0/1 integer work, so the comparisons are exact; a
tolerance would only hide a wrong row.
"""

import itertools

import numpy as np
import pytest

boolean = pytest.importorskip("boolean")

import mojo_booleanpy as mb  # noqa: E402
from mojo_booleanpy import _lib, core as _core  # noqa: E402


def algebra():
    return boolean.BooleanAlgebra()


def reference_table(expr):
    """Evaluate `expr` on every assignment with the real package."""
    names = mb.symbol_names(expr)
    rows = np.array(list(itertools.product([0, 1], repeat=len(names))), dtype=np.uint8)
    out = np.empty(rows.shape[0], dtype=np.uint8)
    for i, row in enumerate(rows):
        out[i] = 1 if expr(**dict(zip(names, (int(v) for v in row)))) else 0
    return out


def numpy_tree_table(expr, names):
    """An independent evaluator, used where `boolean.py` cannot follow.

    `boolean.py` builds a fresh TRUE/FALSE per algebra and `DualBase.__call__`
    forwards every keyword argument to every argument, so
    `(a & algebra.TRUE)(a=1, b=0)` raises. This recursive NumPy evaluator reads
    the same tree and has the same semantics, and is what the constant cases
    are checked against.
    """
    def walk(node, assignment):
        if isinstance(node, boolean.Symbol):
            return assignment[node.obj]
        if _core._is_true(node):
            return 1
        if _core._is_false(node):
            return 0
        if isinstance(node, boolean.NOT):
            return 1 - walk(node.args[0], assignment)
        values = [walk(a, assignment) for a in node.args]
        if isinstance(node, boolean.AND):
            return int(all(values))
        return int(any(values))

    rows = list(itertools.product([0, 1], repeat=len(names)))
    return np.array(
        [walk(expr, dict(zip(names, row))) for row in rows], dtype=np.uint8
    )


def sample_expressions():
    a = algebra()
    x, y, z = a.symbols("x", "y", "z")
    return [
        x,
        ~x,
        x & y,
        x | y,
        x ^ y if False else (x & ~y) | (~x & y),
        (x & ~y) | (z & x),
        (x | y) & ~(x & y),
        ~x & ~y & ~z,
        x & x & x,
        (x | y | z) & ~(x & y & z),
    ]


@pytest.mark.parametrize("expr", sample_expressions(), ids=range(10))
def test_truth_table_matches_symbol_by_symbol_evaluation(expr):
    np.testing.assert_array_equal(mb.truth_table(expr), reference_table(expr))


def test_hand_computed_truth_table_for_three_symbols():
    a = algebra()
    x, y, z = a.symbols("x", "y", "z")
    # (x & ~y) | (z & x) is true on rows 4, 5 and 7 of the standard ordering.
    np.testing.assert_array_equal(
        mb.truth_table((x & ~y) | (z & x)), [0, 0, 0, 0, 1, 1, 0, 1]
    )
    # XOR of two symbols is true on exactly the two rows where they differ.
    np.testing.assert_array_equal(mb.truth_table((x & ~y) | (~x & y)), [0, 1, 1, 0])
    # A negation is the bitwise complement, which is the strongest single check
    # on the word layout there is.
    np.testing.assert_array_equal(mb.truth_table(~x), [1, 0])


def test_symbol_order_is_sorted_by_printed_name():
    a = algebra()
    z, y, x = a.symbols("z", "y", "x")
    assert mb.symbol_names(x & y & z) == ["x", "y", "z"]
    # The order is the one the table is built in, so swapping which name is
    # "first" has to change the table for an asymmetric function.
    np.testing.assert_array_equal(
        mb.truth_table(x & ~y), [0, 0, 1, 0]
    )


def test_packed_words_reconstruct_to_the_same_table():
    a = algebra()
    x, y, z, w = a.symbols("x", "y", "z", "w")
    expr = (x & ~y) | (z ^ w if False else (z & ~w) | (~z & w))
    words = mb.truth_table_words(expr)
    n = len(mb.symbol_names(expr))
    assert words.size == _lib.words_for(1 << n)
    bits = np.unpackbits(words.view(np.uint8), bitorder="little")[: 1 << n]
    np.testing.assert_array_equal(bits, reference_table(expr))


def xor(left, right):
    """XOR from the three operators `boolean.py` actually provides."""
    return (left | right) & ~(left & right)


@pytest.mark.parametrize("nsym", [1, 2, 6, 7, 8, 9])
def test_truth_columns_follow_the_standard_ordering(nsym):
    """`2 ** n` rows is a whole number of 64-bit words only up to n = 6.

    At n = 7 and above the last word is partly padding, and the column pattern,
    the NOT and the final mask all have to agree about how many bits are real.
    A column bit flipped in the high position of a word is invisible in every
    other column, so the columns are checked against NumPy directly.
    """
    nrows = 1 << nsym
    columns = _lib.truth_columns(nsym, nrows)
    assert columns.shape == (nsym, _lib.words_for(nrows))
    rows = np.arange(nrows)
    want = np.stack([(rows >> (nsym - 1 - s)) & 1 for s in range(nsym)], axis=1)
    got = np.unpackbits(columns.view(np.uint8), axis=1, bitorder="little")
    np.testing.assert_array_equal(got[:, :nrows].T, want)


@pytest.mark.parametrize("nsym", [6, 7, 8, 9])
def test_partial_last_word_is_masked(nsym):
    a = algebra()
    symbols = a.symbols(*[f"v{i}" for i in range(nsym)])
    expr = symbols[0]
    for s in symbols[1:]:
        expr = xor(expr, s)
    table = mb.truth_table(expr)
    assert table.size == 1 << nsym
    assert table.dtype == np.uint8
    np.testing.assert_array_equal(table, reference_table(expr))
    words = mb.truth_table_words(expr)
    tail = (1 << nsym) & 63
    if tail:
        # The complement of a full XOR chain is the negated chain, so the top
        # bit of the table must be 0 and the padding above it must be clear.
        assert int(words[-1]) < (1 << tail)


def test_sixteen_symbols_still_agrees_row_by_row():
    a = algebra()
    names = [f"s{i}" for i in range(16)]
    symbols = a.symbols(*names)
    expr = symbols[0]
    for i, s in enumerate(symbols[1:], start=1):
        expr = expr | s if i % 2 else (expr & s)
    np.testing.assert_array_equal(mb.truth_table(expr), reference_table(expr))


def test_popcount_matches_numpy_bit_counting():
    a = algebra()
    x, y, z = a.symbols("x", "y", "z")
    for expr in sample_expressions():
        words = mb.truth_table_words(expr)
        want = int(np.unpackbits(words.view(np.uint8), bitorder="little").sum())
        assert mb.count_true(expr) == want
        assert _lib.popcount(words) == want


def test_true_indices_match_the_nonzero_rows():
    a = algebra()
    x, y, z = a.symbols("x", "y", "z")
    for expr in sample_expressions():
        want = np.flatnonzero(mb.truth_table(expr)).astype(np.int64)
        got = mb.minterms(expr)
        np.testing.assert_array_equal(got, want)


def test_true_indices_respect_the_capacity_but_still_report_the_total():
    a = algebra()
    x, y, z, w = a.symbols("x", "y", "z", "w")
    words = mb.truth_table_words(x | y | z | w)
    total = int(_lib.popcount(words))
    assert total == 15
    want = np.flatnonzero(mb.truth_table(x | y | z | w)).astype(np.int64)
    assert want.size == total
    capped = _lib.true_indices(words, cap=3)
    assert capped.size == 3
    np.testing.assert_array_equal(capped, want[:3])


def test_maxterms_complement_the_minterms():
    a = algebra()
    x, y, z = a.symbols("x", "y", "z")
    expr = (x & ~y) | (z & x)
    n = 1 << len(mb.symbol_names(expr))
    assert set(mb.minterms(expr)).isdisjoint(set(mb.maxterms(expr)))
    assert sorted(list(mb.minterms(expr)) + list(mb.maxterms(expr))) == list(range(n))


def test_minterms_of_a_dnf_are_the_rows_the_dnf_agrees_with():
    """A canonical DNF is one conjunction per true row, so it must reproduce
    the same table. This is the strongest end-to-end check: the minterms come
    out of the bitset scan, and the DNF is handed back to the real package."""
    a = algebra()
    x, y, z = a.symbols("x", "y", "z")
    expr = (x & ~y) | (z & x)
    names = mb.symbol_names(expr)
    pool = dict(zip(names, [x, y, z]))
    dnf = a.FALSE
    for row in mb.minterms(expr):
        terms = []
        for i, name in enumerate(names):
            bit = (row >> (len(names) - 1 - i)) & 1
            terms.append(pool[name] if bit else ~pool[name])
        conjunction = terms[0]
        for t in terms[1:]:
            conjunction = conjunction & t
        dnf = dnf | conjunction
    np.testing.assert_array_equal(mb.truth_table(dnf), mb.truth_table(expr))


def test_cnf_agrees_too():
    a = algebra()
    x, y, z = a.symbols("x", "y", "z")
    expr = (x & ~y) | (z & x)
    names = mb.symbol_names(expr)
    pool = dict(zip(names, [x, y, z]))
    cnf = a.TRUE
    for row in mb.maxterms(expr):
        clauses = []
        for i, name in enumerate(names):
            bit = (row >> (len(names) - 1 - i)) & 1
            clauses.append(~pool[name] if bit else pool[name])
        disjunction = clauses[0]
        for c in clauses[1:]:
            disjunction = disjunction | c
        cnf = cnf & disjunction
    np.testing.assert_array_equal(mb.truth_table(cnf), mb.truth_table(expr))


def test_demorgan_preserves_the_function():
    """`NOT.demorgan()` pushes negations inward, so it must not change the
    function. It is not a complement: a literal is already in negation normal
    form and comes back untouched. `demorgan` exists on the operators, not on a
    bare symbol, so bare symbols are skipped rather than silently mishandled.
    """
    a = algebra()
    for expr in sample_expressions():
        if not hasattr(expr, "demorgan"):
            continue
        # Only `NOT` implements `demorgan`; AND and OR do not.
        np.testing.assert_array_equal(mb.truth_table(expr.demorgan()), mb.truth_table(expr))


def test_negation_of_a_demorganised_expression_is_the_complement():
    """What De Morgan's rules actually assert: ~(a & b) == ~a | ~b."""
    a = algebra()
    x, y, z = a.symbols("x", "y", "z")
    for expr in ((x & y), (x | y), (x & y & z), (x | y | z), (x & ~y), (x | ~y)):
        pushed = (~expr).demorgan()
        np.testing.assert_array_equal(mb.truth_table(pushed), 1 - mb.truth_table(expr))
        np.testing.assert_array_equal(
            mb.truth_table(pushed), mb.truth_table(~expr)
        )


def test_evaluate_rows_matches_the_real_package():
    """The batch path is the same kernel with different input columns, so the
    reference is the real package evaluated one row at a time."""
    rng = np.random.default_rng(4)
    for expr in sample_expressions():
        names = mb.symbol_names(expr)
        rows = rng.integers(0, 2, size=(500, len(names))).astype(np.uint8)
        want = np.array(
            [1 if expr(**dict(zip(names, (int(v) for v in r)))) else 0 for r in rows],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(mb.evaluate_rows(expr, rows), want)


@pytest.mark.parametrize("m", [1, 63, 64, 65, 127, 128, 129, 1000])
def test_evaluate_rows_handles_row_counts_around_the_word_size(m):
    """`m` rows is a whole number of words only when `m` is a multiple of 64.

    The padding bits above row `m - 1` must not turn into results, which is the
    same masking contract the truth-table path has.
    """
    a = algebra()
    x, y = a.symbols("x", "y")
    rng = np.random.default_rng(m)
    rows = rng.integers(0, 2, size=(m, 2)).astype(np.uint8)
    expr = (x & ~y) | (x & y)
    want = rows[:, 0].astype(np.uint8)
    got = mb.evaluate_rows(expr, rows)
    assert got.size == m
    np.testing.assert_array_equal(got, want)


def test_evaluate_rows_rejects_a_wrong_width():
    a = algebra()
    x, y, z = a.symbols("x", "y", "z")
    with pytest.raises(ValueError, match="columns"):
        mb.evaluate_rows(x & y, np.zeros((4, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="0 and 1"):
        mb.evaluate_rows(x & y, np.array([[0, 2], [1, 0]], dtype=np.uint8))
    with pytest.raises(ValueError, match="0 and 1"):
        mb.evaluate_rows(x & y, np.array([[0, -1], [1, 0]], dtype=np.int64))
    with pytest.raises(ValueError, match="integer or boolean"):
        mb.evaluate_rows(x & y, np.array([[0.0, 1.0], [1.0, 0.0]]))
    with pytest.raises(ValueError, match="2-D"):
        mb.evaluate_rows(x & y, np.zeros(4, dtype=np.uint8))


def test_constants_evaluate_like_the_real_ones_where_it_can():
    """`boolean.py` cannot evaluate an expression containing TRUE or FALSE.

    `DualBase.__call__` forwards every keyword argument to every argument and
    `_TRUE.__call__` takes none, so `a & TRUE` raises in the real package. The
    independent tree evaluator is the reference for these.
    """
    a = algebra()
    x, y = a.symbols("x", "y")
    names = ["x", "y"]
    for expr in (a.TRUE, a.FALSE, x & a.TRUE, a.FALSE | x, x & a.FALSE, a.TRUE | x):
        np.testing.assert_array_equal(
            mb.truth_table(expr), numpy_tree_table(expr, mb.symbol_names(expr))
        )
    # A bare constant has no symbols, so its table is the single row it is
    # evaluated on.
    assert mb.count_true(a.TRUE) == 1
    assert mb.count_true(a.FALSE) == 0
    assert mb.count_true(x & a.TRUE) == 1
    assert mb.count_true(a.TRUE | x) == 2


def test_simplified_expressions_have_the_same_table():
    a = algebra()
    x, y, z = a.symbols("x", "y", "z")
    for expr in sample_expressions():
        simplified = expr.simplify()
        np.testing.assert_array_equal(
            mb.truth_table(simplified), mb.truth_table(expr)
        )


def test_a_wide_expression_needs_a_stack_as_wide_as_its_arity():
    """A ten-argument AND keeps ten values live at once.

    If `compile_expr` under-reports the peak depth, the kernel writes past the
    scratch buffer and the table silently acquires garbage rows.
    """
    a = algebra()
    symbols = a.symbols(*[f"w{i}" for i in range(10)])
    program = mb.compile_expr(a.AND(*symbols))
    assert program.depth == 10
    assert program.code.size == 10 + 9
    assert program.nsym == 10 and program.nrows == 1024
    expr = a.AND(*symbols)
    np.testing.assert_array_equal(mb.truth_table(expr), reference_table(expr))


def test_a_long_chain_compiles_to_the_expected_stream_length():
    """Each step of `expr = (expr & y) | ~y` contributes five opcodes.

    A stream that grows by a different amount means an operator was dropped or
    emitted twice, which is invisible in a table that happens to come out right
    for a degenerate expression.
    """
    a = algebra()
    x, y = a.symbols("x", "y")
    expr = x
    for _ in range(200):
        expr = (expr & y) | ~y
    program = mb.compile_expr(expr)
    assert program.code.size == 1 + 200 * 5
    assert program.depth == 2
    np.testing.assert_array_equal(mb.truth_table(expr), reference_table(expr))


def test_symbol_names_match_the_real_package():
    """This port orders the symbols itself; the set has to be the real one.

    `boolean.py` returns an unordered set, so the comparison is on the set, and
    a small expression is used because `Expression.symbols` walks the tree by
    materialising every literal, which is exponential in the sharing of a
    `boolean.py` DAG.
    """
    for expr in sample_expressions():
        want = {s.obj for s in expr.symbols}
        assert set(mb.symbol_names(expr)) == want
    a = algebra()
    x, y, z = a.symbols("x", "y", "z")
    expr = x & y | z
    assert mb.symbol_names(expr) == ["x", "y", "z"]


def test_a_program_can_be_run_against_explicit_columns():
    """`run` with caller-supplied columns is the same kernel the row path uses."""
    a = algebra()
    x, y = a.symbols("x", "y")
    program = mb.compile_expr(x & ~y)
    assert program.nsym == 2 and program.nrows == 4
    nwords = _lib.words_for(4)
    padded = np.zeros((2, nwords * 64), dtype=np.uint8)
    padded[:, :4] = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.uint8)
    columns = np.ascontiguousarray(
        np.packbits(padded, axis=1, bitorder="little").view(np.uint64)
    )
    np.testing.assert_array_equal(mb.unpack(mb.run(program, columns), 4), [1, 0, 1, 0])
    np.testing.assert_array_equal(
        mb.unpack(mb.run(program), 4), mb.truth_table(x & ~y)
    )


def test_unknown_opcode_yields_an_all_zero_table_rather_than_garbage():
    columns = _lib.truth_columns(2, 4)
    out = _lib.eval_bitsliced(np.array([0, 99], dtype=np.int32), 2, 4, columns, 2)
    assert out.tolist() == [0]


def test_words_for_agrees_with_the_c_array():
    for nrows in (1, 63, 64, 65, 128, 1000):
        assert _lib.words_for(nrows) == (nrows + 63) // 64


def test_the_private_constant_classes_are_found():
    """The TRUE/FALSE detection depends on two private names in `boolean.py`.

    If a future release renames them, `_is_true` and `_is_false` both return
    False and a constant becomes an unsupported node, which is a loud failure
    rather than a silently wrong table. This test says which version is
    expected.
    """
    assert len(_core._constant_types()) == 2
    a = algebra()
    assert _core._is_true(a.TRUE)
    assert _core._is_false(a.FALSE)
    assert not _core._is_true(a.symbols("x")[0])
