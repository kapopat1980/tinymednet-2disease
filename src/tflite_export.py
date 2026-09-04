"""
tflite_export.py -- Produces a genuine deployable INT8 artifact.

The original manuscript reported a 3.7 KB footprint computed as
n_parameters x 1 byte, from a model that was never exported and never run in
integer arithmetic. This script builds the same architecture in Keras, converts
it with full-integer quantization, and then reports only quantities read back
from the exported flatbuffer:

  * serialized .tflite size in bytes (the number that matters for Flash)
  * the operator set the runtime must support, and whether any op fell back
    to float
  * accuracy of the exported INT8 model, measured through the TFLite
    interpreter rather than through the float graph

Tensor-arena size is reported as a lower bound computed from the interpreter's
tensor inventory. The true arena depends on TFLite Micro's offline memory
planner and on the target build, so it is labelled as a bound, not a
measurement. No physical microcontroller was available.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import get_task
from experiment import metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts"
OUT.mkdir(exist_ok=True)
L = tf.keras.layers


class SparseAUC(tf.keras.metrics.AUC):
    """ROC-AUC for a 2-way softmax head with integer labels."""

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true, y_pred[:, 1], sample_weight)


def se_block(x, ch, d, r=4):
    # Fixed-size average pool rather than GlobalAveragePooling1D(keepdims=True):
    # the latter emits SHAPE/PACK/STRIDED_SLICE dynamic-shape ops, which TFLite
    # Micro either rejects or handles with a runtime shape stack.
    h = L.AveragePooling1D(pool_size=d)(x)
    h = L.Conv1D(max(1, ch // r), 1, activation="relu")(h)
    h = L.Conv1D(ch, 1, activation="sigmoid")(h)
    # Broadcast (B,1,C) against (B,D,C). Expanding the descriptor explicitly with
    # UpSampling1D instead costs TILE/SHAPE/PACK ops and triples the graph.
    return L.Multiply()([x, h])


def ds_block(x, cout, d, use_se=True, residual=False):
    idt = x
    h = L.DepthwiseConv1D(3, padding="same", use_bias=False)(x)
    h = L.ReLU()(L.BatchNormalization()(h))
    h = L.Conv1D(cout, 1, use_bias=False)(h)
    h = L.ReLU()(L.BatchNormalization()(h))
    if use_se:
        h = se_block(h, cout, d)
    if residual:
        h = L.Add()([h, idt])
    return h


def build_keras(d, width=16, n_classes=2, batch_size=1):
    # A fixed batch dimension is both the realistic microcontroller configuration
    # (one patient per inference) and what keeps the exported graph statically
    # shaped; a dynamic batch emits SHAPE/PACK/STRIDED_SLICE shape arithmetic.
    inp = L.Input((d,), batch_size=batch_size)
    x = L.Reshape((d, 1))(inp)
    x = L.Conv1D(width, 1, use_bias=False)(x)
    x = L.ReLU()(L.BatchNormalization()(x))
    x = ds_block(x, width * 2, d, residual=False)
    x = ds_block(x, width * 2, d, residual=True)
    x = ds_block(x, width, d, residual=False)
    x = L.Flatten()(L.AveragePooling1D(pool_size=d)(x))
    out = L.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out)


def export_int8(model, Xrep, path):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]

    def rep():
        for i in range(min(300, len(Xrep))):
            yield [Xrep[i:i + 1].astype(np.float32)]

    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    blob = conv.convert()
    Path(path).write_bytes(blob)
    raw_bytes = len(blob)
    try:
        from tensorflow.lite.tools import flatbuffer_utils
        mdl = flatbuffer_utils.read_model(str(path))
        flatbuffer_utils.strip_strings(mdl)
        flatbuffer_utils.write_model(mdl, str(path))
    except Exception as e:
        print("  (name stripping unavailable: %s)" % str(e)[:80])
    return raw_bytes, Path(path).stat().st_size


def tflite_predict(path, X):
    it = tf.lite.Interpreter(model_path=str(path))
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    si, zi = inp["quantization"]
    so, zo = out["quantization"]
    probs = np.zeros((len(X), out["shape"][-1]), np.float32)
    for i in range(len(X)):
        q = np.clip(np.round(X[i:i + 1] / si + zi), -128, 127).astype(np.int8)
        it.set_tensor(inp["index"], q)
        it.invoke()
        probs[i] = (it.get_tensor(out["index"])[0].astype(np.float32) - zo) * so
    return probs


def inspect(path):
    it = tf.lite.Interpreter(model_path=str(path))
    it.allocate_tensors()
    ops = sorted({d["op_name"] for d in it._get_ops_details()})
    tensors = it.get_tensor_details()
    dtypes = {}
    arena = 0
    for t in tensors:
        dt = np.dtype(t["dtype"]).name
        dtypes[dt] = dtypes.get(dt, 0) + 1
        if t["shape"].size:
            arena += int(np.prod(t["shape"])) * np.dtype(t["dtype"]).itemsize
    float_tensors = sum(v for k, v in dtypes.items() if k.startswith("float"))
    return dict(operators=ops, n_operators=len(ops), tensor_dtypes=dtypes,
                float_tensors_remaining=float_tensors,
                arena_lower_bound_bytes=arena,
                file_bytes=Path(path).stat().st_size)


def run(task_name, seed=0, epochs=400):
    t = get_task(task_name)
    X, y, d = t["X"], t["y"], t["X"].shape[1]
    Xtr_r, Xte_r, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    Xtr_r, Xva_r, ytr, yva = train_test_split(Xtr_r, ytr, test_size=0.15, stratify=ytr, random_state=0)
    imp = SimpleImputer(strategy="median").fit(Xtr_r)
    sc = StandardScaler().fit(imp.transform(Xtr_r))
    tf_ = lambda A: sc.transform(imp.transform(A)).astype(np.float32)
    Xtr, Xva, Xte = tf_(Xtr_r), tf_(Xva_r), tf_(Xte_r)

    tf.keras.utils.set_random_seed(seed)
    m = build_keras(d, batch_size=None)      # train with a dynamic batch
    cw = {0: len(ytr) / (2 * (ytr == 0).sum()), 1: len(ytr) / (2 * (ytr == 1).sum())}
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=[SparseAUC(name="auc")])
    m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=64, verbose=0,
          class_weight=cw,
          callbacks=[tf.keras.callbacks.EarlyStopping(
              monitor="val_auc", mode="max", patience=60,
              start_from_epoch=40,          # validation AUC is noisy in the first epochs on
              restore_best_weights=True)])  # cohorts this small; stopping there restores noise

    fp32 = m.predict(Xte, verbose=0)
    export_model = build_keras(d, batch_size=1)   # same weights, static batch of 1
    export_model.set_weights(m.get_weights())
    path = OUT / f"tinymednet_{task_name}_int8.tflite"
    raw_bytes, stripped_bytes = export_int8(export_model, Xtr, path)
    q = tflite_predict(path, Xte)

    keras_params = int(sum(np.prod(w.shape) for w in m.weights))
    info = inspect(path)
    return dict(
        task=task_name,
        keras_params=keras_params,
        fp32=metrics(yte, fp32[:, 1], fp32),
        int8_tflite=metrics(yte, q[:, 1], q),
        argmax_agreement=float((fp32.argmax(1) == q.argmax(1)).mean()),
        deployment=info,
        tflite_bytes_with_names=raw_bytes,
        tflite_bytes_stripped=stripped_bytes,
        param_bytes_fp32=keras_params * 4,
        param_bytes_int8=keras_params * 1,
        n_test=int(len(yte)),
    )


if __name__ == "__main__":
    out = {}
    for tn in ["TASK-DIA", "TASK-CKD"]:
        print(f"[{tn}] keras train + tflite int8 export", flush=True)
        out[tn] = run(tn)
        r = out[tn]
        print(f"  params={r['keras_params']}  tflite={r['deployment']['file_bytes']}B  "
              f"FP32 F1={r['fp32']['f1_pos']:.4f}  INT8 F1={r['int8_tflite']['f1_pos']:.4f}  "
              f"agree={r['argmax_agreement']:.3f}", flush=True)
    with open(ROOT / "results" / "tflite_export.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("written results/tflite_export.json")
