#!/usr/bin/env python3
"""Amostrador de potência e memória, em processo separado (não disputa o GIL do benchmark).

Lê o INA3221 (hwmon) da Orin Nano: tensão (mV) e corrente (mA) dos 3 canais
  1 = VDD_IN (entrada do módulo), 2 = VDD_CPU_GPU_CV, 3 = VDD_SOC
mais MemAvailable (/proc/meminfo) e a carga da GPU (0–1000).
O INA3221 atualiza a cada ~20 ms; amostramos a 50 Hz por padrão.
Fica preso ao CPU 0 e para com SIGTERM/SIGINT.

CSV: t_mono_s,vin_mV,iin_mA,vcg_mV,icg_mA,vsoc_mV,isoc_mA,mem_avail_kB,gpu_load
"""
import argparse
import os
import signal
import time
from pathlib import Path

GPU_LOAD = "/sys/devices/platform/bus@0/17000000.gpu/load"


def ina_dir() -> str:
    for h in Path("/sys/class/hwmon").glob("hwmon*"):
        try:
            if (h / "name").read_text().strip() == "ina3221":
                return str(h)
        except OSError:
            pass
    raise FileNotFoundError("INA3221 não encontrado")


def read_int(fd: int) -> int:
    os.lseek(fd, 0, os.SEEK_SET)
    return int(os.read(fd, 64))


def mem_available_kb(fd: int) -> int:
    os.lseek(fd, 0, os.SEEK_SET)
    data = os.read(fd, 4096)
    i = data.index(b"MemAvailable:")
    return int(data[i + 13:data.index(b"kB", i)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--cpu", type=int, default=0)
    a = ap.parse_args()
    try:
        os.sched_setaffinity(0, {a.cpu})
    except OSError:
        pass

    h = ina_dir()
    fds = [os.open(f"{h}/{k}{ch}_input", os.O_RDONLY)
           for ch in (1, 2, 3) for k in ("in", "curr")]
    mem_fd = os.open("/proc/meminfo", os.O_RDONLY)
    try:
        gpu_fd = os.open(GPU_LOAD, os.O_RDONLY)
    except OSError:
        gpu_fd = None

    stop = False

    def _stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    period = 1.0 / a.hz
    with open(a.out, "w", buffering=1 << 16) as fh:
        fh.write("t_mono_s,vin_mV,iin_mA,vcg_mV,icg_mA,vsoc_mV,isoc_mA,mem_avail_kB,gpu_load\n")
        nxt = time.monotonic()
        while not stop:
            t = time.monotonic()
            vals = [read_int(fd) for fd in fds]
            mem = mem_available_kb(mem_fd)
            load = read_int(gpu_fd) if gpu_fd is not None else -1
            fh.write(f"{t:.6f},{','.join(map(str, vals))},{mem},{load}\n")
            nxt += period
            dt = nxt - time.monotonic()
            if dt > 0:
                time.sleep(dt)
            else:
                nxt = time.monotonic()


if __name__ == "__main__":
    main()
