"""Small Linux AMD64 seccomp filter, installed before untrusted Python executes.

Default deny, native ABI only. The allowlist supports CPython/stdlib function
execution, not spawning, exec, namespaces, network or cross-process control.
Numbers are Linux x86_64 syscall ABI constants; unknown syscalls return EPERM.
This is one layer of containment, not a replacement for namespaces or limits.
"""

from __future__ import annotations

import ctypes
import errno
import os
import platform

CONTRACT = "single-process-seccomp-amd64-v1"

# read/write/open/stat, memory, signals, clocks, directory/temporary-file I/O,
# identity queries, futex and CPython runtime bookkeeping. Intentionally absent:
# clone/fork/vfork/exec, socket, ptrace/process_vm, io_uring, mount/setns/unshare,
# credential/capability changes, seccomp/prctl, bpf and kernel module interfaces.
ALLOWED_SYSCALLS = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    23,
    24,
    25,
    26,
    27,
    28,
    32,
    33,
    35,
    36,
    37,
    38,
    39,
    60,
    63,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
    91,
    95,
    96,
    97,
    98,
    99,
    100,
    102,
    104,
    107,
    108,
    110,
    111,
    115,
    116,
    118,
    119,
    120,
    121,
    124,
    131,
    137,
    138,
    140,
    158,
    186,
    187,
    191,
    192,
    193,
    194,
    195,
    196,
    197,
    198,
    199,
    200,
    201,
    202,
    204,
    217,
    218,
    219,
    221,
    228,
    229,
    230,
    231,
    232,
    233,
    257,
    258,
    259,
    260,
    261,
    262,
    263,
    264,
    265,
    266,
    267,
    268,
    269,
    270,
    271,
    273,
    275,
    277,
    278,
    280,
    281,
    285,
    286,
    287,
    289,
    290,
    291,
    292,
    293,
    295,
    296,
    302,
    306,
    307,
    309,
    318,
    319,
    324,
    326,
    327,
    332,
    334,
    436,
    439,
    441,
)


class Instruction(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_ushort),
        ("jt", ctypes.c_ubyte),
        ("jf", ctypes.c_ubyte),
        ("k", ctypes.c_uint32),
    ]


class Program(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ushort), ("filter", ctypes.POINTER(Instruction))]


def install() -> None:
    """Fail closed; no candidate source or candidate imports may precede this."""
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Single-process worker requires Linux x86_64")
    if len(os.listdir("/proc/self/task")) != 1:
        raise RuntimeError("Candidate interpreter must start single-threaded")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.restype = ctypes.c_int
    # BPF loads seccomp_data.arch (offset 4), then seccomp_data.nr (offset 0).
    # Kill non-native ABIs, including x32, before testing the syscall allowlist.
    instructions = [
        (0x20, 0, 0, 4),
        (0x15, 1, 0, 0xC000003E),
        (0x06, 0, 0, 0x80000000),
        (0x20, 0, 0, 0),
        (0x35, 0, 1, 0x40000000),
        (0x06, 0, 0, 0x80000000),
    ]
    for number in ALLOWED_SYSCALLS:
        instructions.extend([(0x15, 0, 1, number), (0x06, 0, 0, 0x7FFF0000)])
    instructions.append((0x06, 0, 0, 0x00050000 | errno.EPERM))
    array = (Instruction * len(instructions))(*(Instruction(*row) for row in instructions))
    program = Program(len(array), array)
    if libc.prctl(38, 1, 0, 0, 0) or libc.prctl(22, 2, ctypes.byref(program), 0, 0):
        raise OSError(ctypes.get_errno(), "Cannot install worker seccomp policy")
