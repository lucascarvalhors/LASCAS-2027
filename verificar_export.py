import json, glob, os
import numpy as np

MODELOS_ESPERADOS = [
    "tribranch18__mlp", "tribranch18__kan",
    "mobilenetv3_light_mlp_tribranch_18cls",
    "mobilenetv3_light_kan_tribranch_18cls",
    "kan_pure_tribranch_18cls", "mlp_pure_tribranch_18cls",
]
ARQS = ["model.onnx", "calib_input0.npy", "calib_input1.npy",
        "calib_input2.npy", "calib_labels.npy", "ref_logits.npy",
        "artifacts.json"]

ok_global = True
for m in MODELOS_ESPERADOS:
    print(f"\n── {m}")
    d = m
    if not os.path.isdir(d):
        print("  ✗ pasta ausente"); ok_global = False; continue

    # 1. arquivos presentes
    faltando = [a for a in ARQS if not os.path.exists(os.path.join(d, a))]
    if faltando:
        print(f"  ✗ faltando: {faltando}"); ok_global = False; continue
    print("  ✓ todos os arquivos presentes")

    # 2. metadados
    meta = json.load(open(os.path.join(d, "artifacts.json")))
    classes = meta.get("classes")
    n_cls = len(classes) if classes else 0
    print(f"  ✓ run_id={meta['run_id']} | classes={n_cls}")
    if n_cls != 18:
        print("  ✗ artifacts.json sem as 18 classes"); ok_global = False

    # 3. calibração: formas e cobertura de classes
    x0 = np.load(os.path.join(d, "calib_input0.npy"))
    x1 = np.load(os.path.join(d, "calib_input1.npy"))
    x2 = np.load(os.path.join(d, "calib_input2.npy"))
    y  = np.load(os.path.join(d, "calib_labels.npy"))
    ref = np.load(os.path.join(d, "ref_logits.npy"))
    n = len(y)
    shapes_ok = (x0.shape == (n, 2048, 6) and x1.shape == (n, 6, 21)
                 and x2.shape == (n, 36) and ref.shape == (n, 18))
    u, c = np.unique(y, return_counts=True)
    print(f"  {'✓' if shapes_ok else '✗'} shapes: {x0.shape} {x1.shape} "
          f"{x2.shape} ref={ref.shape}")
    print(f"  {'✓' if len(u) == 18 else '✗'} classes no teste: {len(u)} "
          f"(min={c.min()} max={c.max()} por classe)")
    if not shapes_ok or len(u) != 18:
        ok_global = False

    # 4. ONNX executa e bate com o Keras
    try:
        import onnxruntime as ort
        s = ort.InferenceSession(os.path.join(d, "model.onnx"),
                                 providers=["CPUExecutionProvider"])
        names = [i.name for i in s.get_inputs()]
        out = s.run(None, {names[0]: x0, names[1]: x1, names[2]: x2})[0]
        dmax = float(np.abs(out - ref).max())
        agree = float((out.argmax(1) == ref.argmax(1)).mean())
        acc = float((out.argmax(1) == y).mean())
        ok = agree == 1.0 and dmax < 5e-2
        print(f"  {'✓' if ok else '✗'} ONNX vs Keras: |Δlogit|max={dmax:.2e} "
              f"argmax={agree:.4f} acc_no_host={acc:.4f}")
        if not ok:
            ok_global = False
    except Exception as e:
        print(f"  ✗ ONNX não executa: {e}"); ok_global = False

print("\n" + "=" * 60)
print("TUDO OK — pode levar para a Jetson" if ok_global
      else "HÁ FALHAS — veja as linhas com ✗")