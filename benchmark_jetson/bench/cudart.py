"""Wrapper mínimo da CUDA Runtime via ctypes.

A Jetson (JetPack 6 / TensorRT 10.3) não tem pycuda nem cuda-python, e o
polygraphy.cuda não expõe eventos. Aqui fica só o que o benchmark precisa:
memória de device e pinned, cópias, streams e eventos CUDA (tempo de GPU).
"""
from __future__ import annotations

import ctypes

import numpy as np


def _load():
    for name in ("libcudart.so.12", "libcudart.so", "/usr/local/cuda/lib64/libcudart.so.12"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise OSError("libcudart não encontrada")


_rt = _load()
_vp = ctypes.c_void_p
_sz = ctypes.c_size_t
_rt.cudaGetErrorString.restype = ctypes.c_char_p
_rt.cudaGetErrorString.argtypes = [ctypes.c_int]
_rt.cudaMalloc.argtypes = [ctypes.POINTER(_vp), _sz]
_rt.cudaFree.argtypes = [_vp]
_rt.cudaMallocHost.argtypes = [ctypes.POINTER(_vp), _sz]
_rt.cudaFreeHost.argtypes = [_vp]
_rt.cudaMemcpy.argtypes = [_vp, _vp, _sz, ctypes.c_int]
_rt.cudaMemcpyAsync.argtypes = [_vp, _vp, _sz, ctypes.c_int, _vp]
_rt.cudaStreamCreate.argtypes = [ctypes.POINTER(_vp)]
_rt.cudaStreamDestroy.argtypes = [_vp]
_rt.cudaStreamSynchronize.argtypes = [_vp]
_rt.cudaEventCreate.argtypes = [ctypes.POINTER(_vp)]
_rt.cudaEventDestroy.argtypes = [_vp]
_rt.cudaEventRecord.argtypes = [_vp, _vp]
_rt.cudaEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), _vp, _vp]
_rt.cudaMemGetInfo.argtypes = [ctypes.POINTER(_sz), ctypes.POINTER(_sz)]

H2D, D2H = 1, 2


def check(err: int) -> None:
    if err != 0:
        raise RuntimeError(f"CUDA error {err}: {_rt.cudaGetErrorString(err).decode()}")


def malloc(nbytes: int) -> int:
    p = _vp()
    check(_rt.cudaMalloc(ctypes.byref(p), max(int(nbytes), 1)))
    return p.value


def free(ptr: int) -> None:
    check(_rt.cudaFree(_vp(ptr)))


def pinned_empty(shape, dtype) -> tuple[np.ndarray, int]:
    """Array numpy sobre memória pinned (cudaMallocHost). Devolve (array, ponteiro)."""
    dtype = np.dtype(dtype)
    n = int(np.prod(shape))
    nbytes = max(n * dtype.itemsize, 1)
    p = _vp()
    check(_rt.cudaMallocHost(ctypes.byref(p), nbytes))
    buf = (ctypes.c_byte * nbytes).from_address(p.value)
    return np.frombuffer(buf, dtype=dtype, count=n).reshape(shape), p.value


def free_host(ptr: int) -> None:
    check(_rt.cudaFreeHost(_vp(ptr)))


def memcpy_htod(dst_dev: int, src: np.ndarray) -> None:
    src = np.ascontiguousarray(src)
    check(_rt.cudaMemcpy(_vp(dst_dev), _vp(src.ctypes.data), src.nbytes, H2D))


def memcpy_htod_async(dst_dev: int, src: np.ndarray, stream: int) -> None:
    check(_rt.cudaMemcpyAsync(_vp(dst_dev), _vp(src.ctypes.data), src.nbytes, H2D, _vp(stream)))


def memcpy_dtoh_async(dst: np.ndarray, src_dev: int, stream: int) -> None:
    check(_rt.cudaMemcpyAsync(_vp(dst.ctypes.data), _vp(src_dev), dst.nbytes, D2H, _vp(stream)))


def stream_create() -> int:
    s = _vp()
    check(_rt.cudaStreamCreate(ctypes.byref(s)))
    return s.value


def stream_sync(stream: int) -> None:
    check(_rt.cudaStreamSynchronize(_vp(stream)))


def event_create() -> int:
    e = _vp()
    check(_rt.cudaEventCreate(ctypes.byref(e)))
    return e.value


def event_record(event: int, stream: int) -> None:
    check(_rt.cudaEventRecord(_vp(event), _vp(stream)))


def event_elapsed_ms(start: int, end: int) -> float:
    ms = ctypes.c_float()
    check(_rt.cudaEventElapsedTime(ctypes.byref(ms), _vp(start), _vp(end)))
    return ms.value


def mem_get_info() -> tuple[int, int]:
    free_b, total_b = _sz(), _sz()
    check(_rt.cudaMemGetInfo(ctypes.byref(free_b), ctypes.byref(total_b)))
    return free_b.value, total_b.value
