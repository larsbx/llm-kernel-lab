# Homogeneous FP64 CPU batches; E0 timings, at most E1 numerical observations.
# No allocations: caller owns disjoint input/output/workspace buffers.
from std.ffi import external_call
from std.math import isfinite
from mode_gram import F64Ptr, _all_finite, _pack_mode1


def _pack_checked(tensor: F64Ptr, packed: F64Ptr, d0: Int, d1: Int, d2: Int) -> Bool:
    var width = d0 * d2
    for j in range(d1):
        for i in range(d0):
            for k in range(d2):
                var value = tensor[unsafe_offset=(i * d1 + j) * d2 + k]
                if not isfinite(value):
                    return False
                packed[unsafe_offset=j * width + i * d2 + k] = value
    return True


@export
def mode_gram_batch_f64(
    tensor_address: Int, batch: Int, d0: Int, d1: Int, d2: Int, mode: Int,
    input_len: Int, output_address: Int, output_len: Int,
    workspace_address: Int, workspace_len: Int, fused: Int,
) abi("C") -> Int32:
    # Status matches scalar API; 6 = invalid preparation strategy.
    if mode < 0 or mode > 2:
        return 1
    if fused != 0 and fused != 1:
        return 6
    comptime LIMIT = 2147483647
    if batch <= 0 or d0 <= 0 or d1 <= 0 or d2 <= 0:
        return 2
    if batch > LIMIT or d0 > LIMIT or d1 > LIMIT or d2 > LIMIT:
        return 2
    if d0 > LIMIT // d1 or d0 * d1 > LIMIT // d2:
        return 2
    var count = d0 * d1 * d2
    var rows = d0
    if mode == 1:
        rows = d1
    elif mode == 2:
        rows = d2
    if rows > LIMIT // rows or batch > LIMIT // count:
        return 2
    var result_count = rows * rows
    if batch > LIMIT // result_count:
        return 2
    if input_len != batch * count or output_len != batch * result_count:
        return 3
    if tensor_address == 0 or output_address == 0:
        return 3
    if mode == 1:
        if workspace_address == 0 or workspace_len != input_len:
            return 3
    elif workspace_len != 0:
        return 3
    var tensor = F64Ptr(unsafe_from_address=tensor_address)
    var output = F64Ptr(unsafe_from_address=output_address)
    var workspace = F64Ptr(unsafe_from_address=workspace_address)
    # Validate the WHOLE batch before the first GEMM/output write. In mode 1
    # fused validation may modify scratch on failure, but leaves output intact.
    if mode == 1 and fused == 1:
        for b in range(batch):
            if not _pack_checked(tensor.unsafe_offset(b * count), workspace.unsafe_offset(b * count), d0, d1, d2):
                return 4
    else:
        if not _all_finite(tensor, input_len):
            return 4
        if mode == 1:
            for b in range(batch):
                _pack_mode1(tensor.unsafe_offset(b * count), workspace.unsafe_offset(b * count), d0, d1, d2)
    var width = count // rows
    var trans_a = Int32(111)
    var trans_b = Int32(112)
    var leading = width
    if mode == 2:
        trans_a = 112
        trans_b = 111
        leading = rows
    for b in range(batch):
        var matrix = tensor.unsafe_offset(b * count)
        if mode == 1:
            matrix = workspace.unsafe_offset(b * count)
        external_call["cblas_dgemm", NoneType](
            Int32(101), trans_a, trans_b, Int32(rows), Int32(rows), Int32(width),
            Float64(1), matrix, Int32(leading), matrix, Int32(leading),
            Float64(0), output.unsafe_offset(b * result_count), Int32(rows),
        )
    if not _all_finite(output, output_len):
        return 5
    return 0
