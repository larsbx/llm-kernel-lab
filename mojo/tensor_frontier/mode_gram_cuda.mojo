# Bounded direct FP64 contraction prototype; no mathematical authority.
# Raw device-pointer ABI is internal. The checked Python API owns validation.
from std.memory import Pointer, alloc
from std.math import ceildiv
from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceStream

comptime F64Ptr = Pointer[Float64, MutUntrackedOrigin]


def _offset(row: Int, column: Int, d1: Int, d2: Int, mode: Int) -> Int:
    if mode == 0:
        return row * d1 * d2 + column
    elif mode == 1:
        return (column // d2 * d1 + row) * d2 + column % d2
    return column * d2 + row


def _entry(x: F64Ptr, index: Int, d0: Int, d1: Int, d2: Int, mode: Int) -> Float64:
    var rows = d0
    if mode == 1:
        rows = d1
    elif mode == 2:
        rows = d2
    var count = d0 * d1 * d2
    var base = (index // (rows * rows)) * count
    var row = (index // rows) % rows
    var col = index % rows
    var value = Float64(0)
    for k in range(count // rows):
        value += x[unsafe_offset=base + _offset(row, k, d1, d2, mode)] * x[unsafe_offset=base + _offset(col, k, d1, d2, mode)]
    return value


def _gram(x_address: Int64, y_address: Int64, d0: Int32, d1: Int32, d2: Int32, mode: Int32, total: Int32):
    var index = global_idx.x
    if index < Int(total):
        var x = F64Ptr(unsafe_from_address=Int(x_address))
        var y = F64Ptr(unsafe_from_address=Int(y_address))
        y[unsafe_offset=index] = _entry(x, index, Int(d0), Int(d1), Int(d2), Int(mode))


struct Session(Movable):
    var ctx: DeviceContext
    var stream: DeviceStream
    var function: type_of(DeviceContext().compile_function[_gram]())

    def __init__(out self, stream_address: Int, device: Int) raises:
        self.ctx = DeviceContext(device_id=device, api="cuda")
        self.function = self.ctx.compile_function[_gram]()
        self.stream = self.ctx.create_external_stream(Pointer[NoneType, MutUntrackedOrigin](unsafe_from_address=stream_address))


@export
def mode_gram_cuda_create(stream_address: Int, device: Int) abi("C") -> Int:
    if stream_address == 0 or device < 0:
        return 0
    try:
        var session = Session(stream_address, device)
        var ptr = alloc[Session](1)
        ptr.unsafe_write(session^)
        return Int(ptr)
    except:
        return 0


@export
def mode_gram_cuda_destroy(handle: Int) abi("C"):
    if handle != 0:
        var ptr = Pointer[Session, MutUntrackedOrigin](unsafe_from_address=handle)
        ptr.unsafe_deinit_pointee()
        ptr.unsafe_free()


@export
def mode_gram_cuda_launch(handle: Int, x: Int, y: Int, batch: Int, d0: Int, d1: Int, d2: Int, mode: Int) abi("C") -> Int32:
    if mode < 0 or mode > 2:
        return 1
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
    if rows > LIMIT // rows or batch > LIMIT // count or batch > LIMIT // (rows * rows):
        return 2
    # Bound this intentionally simple prototype to 2**28 multiply-add terms.
    if batch * rows * rows > 268435456 // (count // rows):
        return 2
    if handle == 0 or x == 0 or y == 0 or x == y:
        return 3
    try:
        var session = Pointer[Session, MutUntrackedOrigin](unsafe_from_address=handle)
        session[].stream.enqueue_function(session[].function, Int64(x), Int64(y), Int32(d0), Int32(d1), Int32(d2), Int32(mode), Int32(batch * rows * rows), grid_dim=ceildiv(batch * rows * rows, 128), block_dim=128)
        return 0
    except:
        return 7


@export
def mode_gram_cuda_host_probe(x: Int, y: Int, d0: Int, d1: Int, d2: Int, mode: Int, total: Int) abi("C"):
    # Test-only host execution of the indexing/contraction helper. Never a fallback.
    var input = F64Ptr(unsafe_from_address=x)
    var output = F64Ptr(unsafe_from_address=y)
    for index in range(total):
        output[unsafe_offset=index] = _entry(input, index, d0, d1, d2, mode)
