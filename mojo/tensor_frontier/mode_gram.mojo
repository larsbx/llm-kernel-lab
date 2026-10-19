# FP64 CPU oracle/experiment only: E0 performance, E1 replicated observations.
# C ABI: dense C-order input, row-major output; lengths count Float64 elements.
from std.ffi import external_call
from std.memory import Pointer, alloc
from std.math import isfinite

comptime F64Ptr = Pointer[Float64, MutUntrackedOrigin]

@export
def mode_gram_f64(
    tensor_address: Int, d0: Int, d1: Int, d2: Int, mode: Int,
    input_len: Int, output_address: Int, output_len: Int,
) abi("C") -> Int32:
    # Status: 1 mode, 2 dimensions/LP64 limit, 3 buffer lengths/null,
    # 4 nonfinite input, 5 nonfinite output. Caller must provide disjoint
    # valid allocations of the declared lengths (as for any raw C pointer API).
    if mode < 0 or mode > 2:
        return 1
    comptime LIMIT = 2147483647
    if d0 <= 0 or d1 <= 0 or d2 <= 0:
        return 2
    if d0 > LIMIT or d1 > LIMIT or d2 > LIMIT:
        return 2
    # Bound the whole tensor before multiplication, including byte offsets.
    if d0 > LIMIT // d1 or d0 * d1 > LIMIT // d2:
        return 2
    var count = d0 * d1 * d2
    var rows = d0
    if mode == 1:
        rows = d1
    elif mode == 2:
        rows = d2
    if rows > LIMIT // rows:
        return 2
    if input_len != count or output_len != rows * rows:
        return 3
    if tensor_address == 0 or output_address == 0:
        return 3
    var tensor = F64Ptr(unsafe_from_address=tensor_address)
    var output = F64Ptr(unsafe_from_address=output_address)
    for i in range(count):
        if not isfinite(tensor[unsafe_offset=i]):
            return 4

    var width = count // rows
    var matrix = tensor
    var packed = tensor
    # C-order linear address is (i*d1+j)*d2+k.
    # mode 0: row i, column j*d2+k (already contiguous).
    # mode 1: row j, column i*d2+k (pack once in Mojo).
    # mode 2: row k, column i*d1+j; use BLAS transpose, no copy.
    if mode == 1:
        packed = alloc[Float64](count)
        for j in range(d1):
            for i in range(d0):
                for k in range(d2):
                    packed[unsafe_offset=j * width + i * d2 + k] = tensor[unsafe_offset=(i * d1 + j) * d2 + k]
        matrix = packed

    var trans_a = Int32(111)  # CblasNoTrans
    var trans_b = Int32(112)  # CblasTrans
    var leading = width
    if mode == 2:
        trans_a = 112
        trans_b = 111
        leading = rows
    # Existing LP64 CBLAS GEMM; no custom multiply/reduction implementation.
    external_call["cblas_dgemm", NoneType](
        Int32(101), trans_a, trans_b, Int32(rows), Int32(rows), Int32(width),
        Float64(1), matrix, Int32(leading), matrix, Int32(leading),
        Float64(0), output, Int32(rows),
    )
    if mode == 1:
        packed.unsafe_free()
    for i in range(rows * rows):
        if not isfinite(output[unsafe_offset=i]):
            return 5
    return 0
