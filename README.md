# mojo-boolean-py

`mojo-boolean-py` is the compute-oriented subset of
[boolean.py](https://github.com/bastikr/boolean.py) with truth-table
evaluation implemented as a bitsliced Mojo kernel.

`boolean.py` is a symbolic algebra over `{TRUE, FALSE}`: a tokenizer, a
recursive-descent parser, an `Expression` tree of `Symbol` / `NOT` / `AND` /
`OR`, and rules that normalise a tree to DNF or CNF or push negations inward.
Almost all of that is tree rewriting over Python objects and is left alone.
The one place with real arithmetic is *evaluation*, and `boolean.py` does it one
assignment at a time through `expr(**kwargs)`.

A Boolean function of `n` symbols has `2 ** n` independent rows, so the whole
table can be held `n` columns wide and 64 rows to a machine word: one `UInt64`
per symbol column, and `~`, `&`, `|` become word operations. That is the port.

```python
import boolean
import mojo_booleanpy as mb

algebra = boolean.BooleanAlgebra()
a, b = algebra.symbols("a", "b")
mb.truth_table(a & ~b)      # uint8 [0 0 1 0]
mb.minterms(a & ~b)         # int64 [2]
mb.count_true(a | b)        # 3
mb.evaluate_rows(a | b, [[1, 0], [0, 1], [1, 1]])   # uint8 [1 1 1]
```

## Is there a numeric core here?

Mostly not, and this port does not pretend otherwise. Of the 1662 lines in
`boolean.py`, the arithmetic is: the truth table, its complement, the
satisfying-assignment set, and a bitset scan. Everything else is parsing,
hashing, comparison, sorting and `frozenset` algebra. The table below is the
honest accounting.

| `boolean.py` surface | what it does | disposition |
| --- | --- | --- |
| `Expression.__call__` | evaluate one assignment by recursion | **ported** as a bitsliced kernel that does all `2 ** n` at once |
| `Expression.symbols` / `get_literals` / `literals` / `objects` | collect the symbols of a tree | the symbol set is the real package's; the traversal is this port's (see below) |
| `Expression.demorgan`, `cancel` | push negations to the literals | left in `boolean.py`; tested through its output here |
| `BooleanAlgebra.normalize` / `cnf` / `dnf` | distribute, absorb, sort terms into canonical form | left in `boolean.py`: it is `frozenset` algebra, not arithmetic |
| `DualBase.simplify`, `absorb`, `subtract`, `flatten`, `distributive` | term rewriting | left in `boolean.py`, same reason |
| `BooleanAlgebra.parse`, `tokenize`, `_start_operation`, `ParseError` | parsing | left alone |
| `Expression.__eq__`, `__hash__`, `__lt__`, `pretty`, `__str__` | structural identity and rendering | left alone |

The Python package is `mojo_booleanpy`, so it installs alongside the real
`boolean` and the tests import both and compare them.

## Covered subset

| area | implemented API |
| --- | --- |
| Compilation | `compile_expr(expr) -> Program` (`code`, `names`, `depth`, `nsym`, `nrows`) |
| Truth tables | `truth_table(expr)`, `truth_table_words(expr)`, `unpack(words, nrows)`, `run(program, columns=None)` |
| Batch evaluation | `evaluate_rows(expr, assignments)` — the same kernel over an arbitrary (m, n) 0/1 matrix |
| Satisfying sets | `minterms(expr)`, `maxterms(expr)`, `count_true(expr)` |
| Symbols | `symbol_names(expr)` — the real package's set, in a reproducible order |
| Kernels | `bnl_truth_columns`, `bnl_eval_bitsliced`, `bnl_popcount`, `bnl_true_indices`, `bnl_words_for` |

Not implemented: the whole of the second table above. `boolean.py` is the right
tool for all of it, and none of it would benefit from a compiled inner loop.

### Conventions this port had to pin down

* **Symbol order.** `Expression.symbols` is an unordered `set`, so a row index
  into a truth table would not be reproducible. Symbols are ordered by their
  printed name, and row `r` assigns symbol `s` the value
  `(r >> (n - 1 - s)) & 1`. `test_symbol_names_match_the_real_package` checks
  the set against the real property.
* **Constants.** `TRUE` and `FALSE` are per-algebra instances
  (`algebra.TRUE`), and `DualBase.__call__` forwards every keyword argument to
  every argument, so `boolean.py` cannot evaluate an expression that contains
  one: `(a & algebra.TRUE)(a=1, b=0)` raises. This port can, because the
  stream has `OP_TRUE` / `OP_FALSE` opcodes. The tests use an independent
  recursive evaluator as the reference for those cases and say why.
* **Tree traversal.** `Expression.symbols` reaches its answer through
  `get_literals`, which materialises a list of every literal in the tree. Since
  `boolean.py` expressions are DAGs with sharing — `a ^ b` written as
  `(a | b) & ~(a & b)` mentions `a` and `b` twice — any recursive walk costs
  `2 ** k` for a chain of `k` shared gates. `symbol_names` walks the tree
  itself with an explicit stack, which is the same set for linear time. This
  is not a kernel, it is a traversal the kernel needs, and it is why
  `compile_expr` is worth doing once and keeping.

## Install

```bash
bash build/build.sh          # -> dist/libmojo-boolean-py.so
PYTHONPATH=python python -m pytest tests -q
```

Set `PYTHONPATH=python` when using the package outside a Pixi task.

## Performance

Best-of-three wall clock on a shared box, against a fully vectorised NumPy
evaluator of the same expression tree. Every case verifies the answer before
timing. The figures are small in absolute terms — a 14-symbol truth table is
16384 rows, which is microseconds of work — so the ratios matter more than the
milliseconds, and run-to-run variation on this machine is large.

| case | reference | mojo-boolean-py | result |
| --- | ---: | ---: | ---: |
| truth table, n=14, 2^14 rows | 0.08 ms | 0.05 ms | 1.6x faster |
| truth-table columns, n=14 | 1.51 ms | 0.54 ms | 2.8x faster |
| `evaluate_rows`, m=1048576 n=12 | 15.3 ms | 27.3 ms | 0.56x, slower |
| `minterms`, n=18, 2^18 rows | 94.1 ms | 30.7 ms | 3.1x faster |

The truth-table cases win because the packed form does one 64-bit operation
where the byte form does 64 byte operations, and because the minterm scan reads
262144 bits instead of 262144 bytes. The numbers are far from 64x because at
this size the ctypes call, the `np.unpackbits` on the way out and NumPy's
already-vectorised baseline dominate.

`evaluate_rows` is a loss and is reported as one. A uint8 NumPy baseline is
already 32-wide under AVX2, so bitslicing buys a factor of two on the
operators, and the pack and unpack round trip costs two extra passes over 12 MB
of input. Bitslicing pays when the table is the answer; it does not pay when
the answer is a byte array of results over given rows.

Reproduce with:

```bash
python bench/bench.py
```

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit.
`build/build.sh` compiles it with `mojo build --emit shared-lib` into
`dist/libmojo-boolean-py.so`.

The Python layer compiles the `boolean.py` tree into a postfix stream once and
then makes one call per evaluation. Buffers cross the C ABI as 64-bit addresses
and are reconstructed in Mojo as `Pointer[UInt64, AnyOrigin[mut=True]]`, which
keeps the exported symbols non-parametric. The kernel keeps an explicit stack
in caller-owned scratch of `depth * nwords` words; `depth` is what
`compile_expr` reports and is the reason a caller should compile once.

Two details the kernel has to get right and the tests pin down:

* **Binary operators are emitted one at a time.** `boolean.py` lets `AND` and
  `OR` take any number of arguments and folds them left to right, and a single
  n-ary opcode would leave the kernel's stack pointer no longer matching the
  expression, so the final result would be read from a stale row. `test_a_long_chain_compiles_to_the_expected_stream_length`
  counts the opcodes to keep that honest.
* **The final word is masked.** `2 ** n` rows is a whole number of 64-bit words
  only up to `n = 6`, so above that the last word is partly padding. The
  evaluator clears the padding, and `maxterms`, which complements a packed
  table, has to clear it again or every position past the end of the table
  comes back as a maxterm.

Everything here is 0/1 integer work, so no parity test in this repository uses
a tolerance: `np.array_equal` is the right tool, and it is what the tests use.

## Tests

51 tests, all against the real `boolean` package or an independent evaluator
where `boolean.py` cannot follow.

## License

MIT, matching the repository LICENSE.
