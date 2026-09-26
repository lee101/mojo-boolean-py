"""Bitsliced evaluation kernels for the symbolic algebra in `boolean.py`.

`boolean.py` is a symbolic Boolean algebra: a tokenizer, a parser, an
expression tree, and rules that normalise a tree to DNF or CNF. Almost all of
it is tree rewriting over Python objects. The one place where real arithmetic
lives is *evaluation*, and the honest way to make that fast is to stop
evaluating one assignment at a time.

A Boolean function of `n` symbols has `2 ** n` rows in its truth table, and
every row is independent, so the whole table can be held `n` columns wide and
64 rows to a machine word: one `UInt64` per symbol column, and `~`, `&`, `|`
become word operations. That is a factor of 64 less work than a byte array,
and it is the whole of `bnl_eval_bitsliced` below.

Every exported symbol takes buffer addresses as plain `Int` values and rebuilds
the pointer inside the body, because `@export` rejects parametric functions and
an inferred pointer origin would make the symbol parametric.
"""

from std.bit import pop_count
from std.math import log2

comptime UPtr = Pointer[UInt64, AnyOrigin[mut=True]]
comptime I32Ptr = Pointer[Int32, AnyOrigin[mut=True]]
comptime I64Ptr = Pointer[Int64, AnyOrigin[mut=True]]

# Postfix opcodes. A non-negative value is a symbol index; the negatives are the
# three operators `boolean.py` implements.
comptime OP_NOT: Int32 = -1
comptime OP_AND: Int32 = -2
comptime OP_OR: Int32 = -3
# `boolean.py` carries TRUE and FALSE as ordinary elements of the algebra, and
# `Expression.simplify()` produces them, so the stream has to be able to name
# them rather than leaving the caller to peel them off by hand.
comptime OP_TRUE: Int32 = -4
comptime OP_FALSE: Int32 = -5


def up(addr: Int) -> UPtr:
    return UPtr(unsafe_from_address=addr)


def i32p(addr: Int) -> I32Ptr:
    return I32Ptr(unsafe_from_address=addr)


def i64p(addr: Int) -> I64Ptr:
    return I64Ptr(unsafe_from_address=addr)


@export("bnl_truth_columns")
def bnl_truth_columns(
    nsym: Int, nrows: Int, nwords: Int, out_addr: Int
) abi("C"):
    """Write the standard truth-table columns, one bit-packed word per block.

    Row `r` of the `nrows = 2 ** nsym` table assigns symbol `s` the value
    `(r >> (nsym - 1 - s)) & 1`, so column 0 is a block of zeros followed by a
    block of ones, the next column two blocks each, and so on. Row `r` of the
    output lives in word `r / 64` at bit `r % 64`, which is the layout
    `np.packbits(..., bitorder="little")` produces.

    The per-bit inner loop is deliberate. The column is a run-length pattern,
    and any closed form for it has to special-case runs of exactly 64 bits and
    the shift-by-64 boundary; this kernel runs once per expression while the
    evaluator below runs `2 ** nsym / 64` times per operator, so clarity here
    is worth more than the factor it gives up.
    """
    var out = up(out_addr)
    for s in range(nsym):
        var shift = nsym - 1 - s
        var base = s * nwords
        for w in range(nwords):
            var word = UInt64(0)
            for b in range(64):
                var r = w * 64 + b
                if r < nrows and ((r >> shift) & 1) == 1:
                    word |= UInt64(1) << UInt64(b)
            out[unsafe_offset=base + w] = word


@export("bnl_eval_bitsliced")
def bnl_eval_bitsliced(
    code_addr: Int,
    nops: Int,
    nsym: Int,
    nrows: Int,
    nwords: Int,
    col_addr: Int,
    stack_addr: Int,
    out_addr: Int,
) abi("C"):
    """Evaluate a NOT/AND/OR expression with 64 table rows per machine word.

    `code` is the postfix stream: a value in `0 .. nsym - 1` pushes symbol
    column `v`, `OP_TRUE` / `OP_FALSE` push a constant, `OP_NOT` complements
    the top of the stack, and `OP_AND` / `OP_OR` pop two and push their bitwise
    combination. `col` holds `nsym` columns of `nwords` words each, already
    bit-packed by `bnl_truth_columns` or by the caller.

    `stack` is caller-owned scratch for `depth * nwords` words, where `depth`
    is the deepest the postfix evaluation ever reaches. `out` receives the
    result, with the bits beyond row `nrows` of the final word cleared, so the
    caller never has to reason about padding.
    """
    var code = i32p(code_addr)
    var col = up(col_addr)
    var stack = up(stack_addr)
    var out = up(out_addr)

    var sp = 0
    for k in range(nops):
        var op = code[unsafe_offset=k]
        if op >= 0 and Int(op) < nsym:
            for w in range(nwords):
                stack[unsafe_offset=sp * nwords + w] = col[unsafe_offset=Int(op) * nwords + w]
            sp += 1
        elif op == OP_TRUE or op == OP_FALSE:
            var fill = ~UInt64(0) if op == OP_TRUE else UInt64(0)
            for w in range(nwords):
                stack[unsafe_offset=sp * nwords + w] = fill
            sp += 1
        elif op == OP_NOT:
            var row = (sp - 1) * nwords
            for w in range(nwords):
                stack[unsafe_offset=row + w] = ~stack[unsafe_offset=row + w]
        elif op == OP_AND:
            # The operands sat at rows `sp` and `sp - 1`; the result goes back
            # to `sp - 1` once the stack is popped.
            sp -= 1
            var a = (sp - 1) * nwords
            var b = sp * nwords
            for w in range(nwords):
                stack[unsafe_offset=a + w] = stack[unsafe_offset=a + w] & stack[unsafe_offset=b + w]
        elif op == OP_OR:
            sp -= 1
            var a = (sp - 1) * nwords
            var b = sp * nwords
            for w in range(nwords):
                stack[unsafe_offset=a + w] = stack[unsafe_offset=a + w] | stack[unsafe_offset=b + w]
        else:
            # An unknown opcode would leave the stack balanced by accident, so
            # zero the result instead of reporting whatever happened to be left.
            for w in range(nwords):
                out[unsafe_offset=w] = UInt64(0)
            return

    var top = (sp - 1) * nwords
    for w in range(nwords):
        out[unsafe_offset=w] = stack[unsafe_offset=top + w]
    var tail = nrows & 63
    if tail != 0 and nwords > 0:
        out[unsafe_offset=nwords - 1] &= (UInt64(1) << UInt64(tail)) - UInt64(1)


@export("bnl_popcount")
def bnl_popcount(words_addr: Int, nwords: Int, out_addr: Int) abi("C"):
    """Number of set bits across a packed truth table.

    The size of the satisfying set of a Boolean function, which is what a
    counting or model-checking use of this package actually wants.
    """
    var words = up(words_addr)
    var out = i64p(out_addr)
    var total = Int64(0)
    for w in range(nwords):
        total += Int64(pop_count(words[unsafe_offset=w]))
    out[unsafe_offset=0] = total


@export("bnl_true_indices")
def bnl_true_indices(
    words_addr: Int, nwords: Int, out_addr: Int, cap: Int, count_addr: Int
) abi("C"):
    """Row indices of the set bits of a packed truth table.

    The rows a function is true on are its minterms. `out` receives up to `cap`
    of them in ascending order and `count` always receives the true total, so a
    caller can size the buffer first and then take the rows it wants.
    """
    var words = up(words_addr)
    var out = i64p(out_addr)
    var count = i64p(count_addr)
    var k = 0
    for w in range(nwords):
        var word = words[unsafe_offset=w]
        if word == UInt64(0):
            continue
        for b in range(64):
            if ((word >> UInt64(b)) & UInt64(1)) != UInt64(0):
                if k < cap:
                    out[unsafe_offset=k] = Int64(w * 64 + b)
                k += 1
    count[unsafe_offset=0] = Int64(k)


@export("bnl_words_for")
def bnl_words_for(nrows: Int) abi("C") -> Int:
    """Words needed to hold `nrows` bits: `(nrows + 63) // 64`.

    Exposed so the Python side and the kernels cannot disagree about the size
    of a packed truth table.
    """
    return (nrows + 63) // 64
