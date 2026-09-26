"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a 64-bit
address, so the argtypes below must stay `c_int64` for addresses; `c_int`
truncates them and segfaults.
"""

import ctypes
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-boolean-py.so"

I32 = ctypes.c_int32
I64 = ctypes.c_int64
ADDR = ctypes.c_int64


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.bnl_truth_columns.restype = None
    lib.bnl_truth_columns.argtypes = [I64, I64, I64, ADDR]
    lib.bnl_eval_bitsliced.restype = None
    lib.bnl_eval_bitsliced.argtypes = [ADDR, I64, I64, I64, I64, ADDR, ADDR, ADDR]
    lib.bnl_popcount.restype = None
    lib.bnl_popcount.argtypes = [ADDR, I64, ADDR]
    lib.bnl_true_indices.restype = None
    lib.bnl_true_indices.argtypes = [ADDR, I64, ADDR, I64, ADDR]
    lib.bnl_words_for.restype = I64
    lib.bnl_words_for.argtypes = [I64]
    return lib


lib = _load()

OP_NOT = -1
OP_AND = -2
OP_OR = -3
OP_TRUE = -4
OP_FALSE = -5


def _addr(a: np.ndarray) -> int:
    return a.ctypes.data


def words_for(nrows: int) -> int:
    return int(lib.bnl_words_for(int(nrows)))


def truth_columns(nsym: int, nrows: int | None = None) -> np.ndarray:
    """Packed standard truth-table columns, shape (nsym, nwords) uint64.

    Bit `r` of column `s` is `(r >> (nsym - 1 - s)) & 1`, so unpacking the
    result reproduces the usual ordering where the first symbol varies fastest
    within each block of `2 ** (nsym - 1 - s)` rows.
    """
    if nrows is None:
        nrows = 1 << nsym
    nwords = words_for(nrows)
    out = np.zeros(nsym * nwords, dtype=np.uint64)
    lib.bnl_truth_columns(nsym, nrows, nwords, _addr(out))
    return out.reshape(nsym, nwords)


def eval_bitsliced(code, nsym, nrows, columns, depth) -> np.ndarray:
    """Evaluate a postfix stream and return the packed result words.

    `code` is an int32 array of symbol indices and `OP_*` values, `columns` is
    an (nsym, nwords) uint64 array of bit-packed input columns, and `depth` is
    the stack the postfix evaluation needs.
    """
    code = np.ascontiguousarray(code, dtype=np.int32)
    columns = np.ascontiguousarray(columns, dtype=np.uint64)
    nrows = int(nrows)
    nwords = columns.shape[1]
    stack = np.zeros(depth * nwords, dtype=np.uint64)
    out = np.zeros(nwords, dtype=np.uint64)
    lib.bnl_eval_bitsliced(
        _addr(code), code.size, nsym, nrows, nwords,
        _addr(columns), _addr(stack), _addr(out),
    )
    return out


def popcount(words) -> int:
    words = np.ascontiguousarray(words, dtype=np.uint64)
    out = np.zeros(1, dtype=np.int64)
    lib.bnl_popcount(_addr(words), words.size, _addr(out))
    return int(out[0])


def true_indices(words, cap: int | None = None) -> np.ndarray:
    """Row indices of the set bits, ascending. `cap` truncates the result."""
    words = np.ascontiguousarray(words, dtype=np.uint64)
    if cap is None:
        cap = int(words.size) * 64
    out = np.zeros(max(cap, 1), dtype=np.int64)
    count = np.zeros(1, dtype=np.int64)
    lib.bnl_true_indices(_addr(words), words.size, _addr(out), cap, _addr(count))
    return out[: int(count[0])]
