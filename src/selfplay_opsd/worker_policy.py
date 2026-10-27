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
ALLOWED_SYSCALLS = {
    "read": 0,
    "write": 1,
    "open": 2,
    "close": 3,
    "stat": 4,
    "fstat": 5,
    "lstat": 6,
    "poll": 7,
    "lseek": 8,
    "mmap": 9,
    "mprotect": 10,
    "munmap": 11,
    "brk": 12,
    "rt_sigaction": 13,
    "rt_sigprocmask": 14,
    "rt_sigreturn": 15,
    "pread64": 17,
    "pwrite64": 18,
    "readv": 19,
    "writev": 20,
    "access": 21,
    "select": 23,
    "sched_yield": 24,
    "mremap": 25,
    "msync": 26,
    "mincore": 27,
    "madvise": 28,
    "dup": 32,
    "dup2": 33,
    "nanosleep": 35,
    "getitimer": 36,
    "alarm": 37,
    "setitimer": 38,
    "getpid": 39,
    "exit": 60,
    "uname": 63,
    "fcntl": 72,
    "flock": 73,
    "fsync": 74,
    "fdatasync": 75,
    "truncate": 76,
    "ftruncate": 77,
    "getdents": 78,
    "getcwd": 79,
    "chdir": 80,
    "fchdir": 81,
    "rename": 82,
    "mkdir": 83,
    "rmdir": 84,
    "creat": 85,
    "link": 86,
    "unlink": 87,
    "symlink": 88,
    "readlink": 89,
    "chmod": 90,
    "fchmod": 91,
    "umask": 95,
    "gettimeofday": 96,
    "getrlimit": 97,
    "getrusage": 98,
    "sysinfo": 99,
    "times": 100,
    "getuid": 102,
    "getgid": 104,
    "geteuid": 107,
    "getegid": 108,
    "getppid": 110,
    "getpgrp": 111,
    "getgroups": 115,
    "getresuid": 118,
    "getresgid": 120,
    "getpgid": 121,
    "getsid": 124,
    "sigaltstack": 131,
    "statfs": 137,
    "fstatfs": 138,
    "getpriority": 140,
    "arch_prctl": 158,
    "gettid": 186,
    "readahead": 187,
    "getxattr": 191,
    "lgetxattr": 192,
    "fgetxattr": 193,
    "listxattr": 194,
    "llistxattr": 195,
    "flistxattr": 196,
    "removexattr": 197,
    "lremovexattr": 198,
    "fremovexattr": 199,
    "tkill": 200,
    "time": 201,
    "futex": 202,
    "sched_getaffinity": 204,
    "getdents64": 217,
    "set_tid_address": 218,
    "restart_syscall": 219,
    "fadvise64": 221,
    "clock_gettime": 228,
    "clock_getres": 229,
    "clock_nanosleep": 230,
    "exit_group": 231,
    "epoll_wait": 232,
    "epoll_ctl": 233,
    "openat": 257,
    "mkdirat": 258,
    "futimesat": 261,
    "newfstatat": 262,
    "unlinkat": 263,
    "renameat": 264,
    "linkat": 265,
    "symlinkat": 266,
    "readlinkat": 267,
    "fchmodat": 268,
    "faccessat": 269,
    "pselect6": 270,
    "ppoll": 271,
    "set_robust_list": 273,
    "splice": 275,
    "sync_file_range": 277,
    "vmsplice": 278,
    "utimensat": 280,
    "epoll_pwait": 281,
    "fallocate": 285,
    "timerfd_settime": 286,
    "timerfd_gettime": 287,
    "signalfd4": 289,
    "eventfd2": 290,
    "epoll_create1": 291,
    "dup3": 292,
    "pipe2": 293,
    "preadv": 295,
    "pwritev": 296,
    "prlimit64": 302,
    "syncfs": 306,
    "getcpu": 309,
    "getrandom": 318,
    "memfd_create": 319,
    "membarrier": 324,
    "copy_file_range": 326,
    "preadv2": 327,
    "statx": 332,
    "rseq": 334,
    "close_range": 436,
    "faccessat2": 439,
    "epoll_pwait2": 441,
}


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
    for number in ALLOWED_SYSCALLS.values():
        instructions.extend([(0x15, 0, 1, number), (0x06, 0, 0, 0x7FFF0000)])
    instructions.append((0x06, 0, 0, 0x00050000 | errno.EPERM))
    array = (Instruction * len(instructions))(*(Instruction(*row) for row in instructions))
    program = Program(len(array), array)
    if libc.prctl(38, 1, 0, 0, 0) or libc.prctl(22, 2, ctypes.byref(program), 0, 0):
        raise OSError(ctypes.get_errno(), "Cannot install worker seccomp policy")
