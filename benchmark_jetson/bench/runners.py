"""Runners de inferência com batch = 1.

Cada `infer(feed)` é síncrono: copia as entradas para a GPU, executa, copia a saída de
volta e espera terminar. Devolve (saída, tempo de GPU em ms). O tempo de GPU vem de
eventos CUDA em volta do `execute_async_v3` (TensorRT); para o ONNX Runtime é NaN.
O array devolvido é um buffer reutilizado: copie-o se for guardar.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import common as C
import cudart


class TrtRunner:
    def __init__(self, plan_path: Path):
        import tensorrt as trt
        self.backend = "tensorrt"
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        plan = Path(plan_path).read_bytes()
        self.engine_bytes = len(plan)
        self.engine = self.runtime.deserialize_cuda_engine(plan)
        if self.engine is None:
            raise RuntimeError(f"falha ao desserializar {plan_path}")
        self.ctx = self.engine.create_execution_context()
        self.stream = cudart.stream_create()
        self.ev0, self.ev1 = cudart.event_create(), cudart.event_create()
        self.inputs, self.outputs = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                shape = tuple(self.engine.get_tensor_shape(name))
                if any(s < 0 for s in shape):
                    raise RuntimeError(f"engine com forma dinâmica em {name}: {shape}")
                self.inputs.append(name)
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
                self.outputs.append(name)
        # Saída principal = logits (1, N_CLASSES). As demais são auxiliares (contorno do defeito do
        # Myelin no mlp_pure, ver build_engines.AUX_OUTPUTS): têm buffer na GPU, mas não voltam ao host.
        main = [n for n in self.outputs if tuple(self.engine.get_tensor_shape(n)) == (1, C.N_CLASSES)]
        if len(main) != 1:
            raise RuntimeError(f"esperava 1 saída de logits, achei {self.outputs}")
        self.aux_outputs = [n for n in self.outputs if n != main[0]]
        self.outputs = main
        self.aux_dev = []
        for name in self.aux_outputs:
            nbytes = int(np.prod(tuple(self.ctx.get_tensor_shape(name)))) * \
                np.dtype(trt.nptype(self.engine.get_tensor_dtype(name))).itemsize
            dev = cudart.malloc(nbytes)
            self.ctx.set_tensor_address(name, dev)
            self.aux_dev.append(dev)
        self.buf = {}
        for name in self.inputs + self.outputs:
            shape = tuple(self.ctx.get_tensor_shape(name))
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            host, host_ptr = cudart.pinned_empty(shape, dtype)
            dev = cudart.malloc(host.nbytes)
            self.ctx.set_tensor_address(name, dev)
            self.buf[name] = (host, dev)
        dm = getattr(self.engine, "device_memory_size_v2", None)
        self.device_memory_bytes = int(dm if dm is not None else self.engine.device_memory_size)
        self.input_names = list(self.inputs)

    def infer(self, feed: dict) -> tuple[np.ndarray, float]:
        s = self.stream
        for name in self.inputs:
            host, dev = self.buf[name]
            np.copyto(host, feed[name])
            cudart.memcpy_htod_async(dev, host, s)
        cudart.event_record(self.ev0, s)
        if not self.ctx.execute_async_v3(s):
            raise RuntimeError("execute_async_v3 falhou")
        cudart.event_record(self.ev1, s)
        host, dev = self.buf[self.outputs[0]]
        cudart.memcpy_dtoh_async(host, dev, s)
        cudart.stream_sync(s)
        return host, cudart.event_elapsed_ms(self.ev0, self.ev1)


class OrtRunner:
    def __init__(self, onnx_path: Path, device: str):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        if device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif device == "cuda":
            providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        else:
            raise ValueError(device)
        self.backend = f"onnxruntime-{device}"
        self.sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers)
        self.providers = self.sess.get_providers()
        if device == "cuda" and "CUDAExecutionProvider" not in self.providers:
            raise RuntimeError(f"CUDA EP não ativo: {self.providers}")
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.engine_bytes = None
        self.device_memory_bytes = None

    def infer(self, feed: dict) -> tuple[np.ndarray, float]:
        out = self.sess.run(None, {n: feed[n] for n in self.input_names})[0]
        return out, float("nan")
