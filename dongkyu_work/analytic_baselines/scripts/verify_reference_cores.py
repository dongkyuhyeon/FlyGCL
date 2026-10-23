"""CPU reference-core checks. These do NOT exercise FlyGCL or an image backbone.

Run from any directory. Requires torch; no dataset, checkpoint, or network.
Reference files remain byte-identical. SLDA's hard-coded CUDA device is changed
to CPU only in the in-memory AST used by this test. GACL classes are extracted
from their AST to bypass imports of the original dataset/trainer framework.
"""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
torch.manual_seed(23)


def check_sources():
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())
    for f in manifest["files"]:
        data = (ROOT / f["path"]).read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        assert actual == f["git_blob_sha"], f["path"]
    print(f"PASS exact upstream blob hashes: {len(manifest['files'])} files")


def load_class(path, name, cpu_init=False):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    replacements = 0
    if cpu_init:
        for node in ast.walk(cls):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and node.value.value == "cuda":
                assert len(node.targets) == 1 and ast.unparse(node.targets[0]) == "self.device"
                node.value = ast.Constant(value="cpu")
                replacements += 1
        assert replacements == 1
    namespace = {"torch": torch, "nn": nn, "F": F, "os": os, "_Trainer": object}
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    exec(compile(module, path, "exec"), namespace)
    return namespace[name]


def check_gacl():
    cls = load_class("original/gacl/methods/GACL.py", "GACL")
    x = torch.randn(19, 7, dtype=torch.double)
    # A class reappears after another arrives; the final class appears late.
    y = torch.tensor([0, 0, 0, 1, 1, 0, 1, 0, 0, 2, 2, 1, 2, 0, 2, 1, 1, 0, 2])
    gamma = 0.7
    for chunks in ([19], [1] * 19, [3, 6, 2, 8]):
        head = cls.__new__(cls)
        head.model = SimpleNamespace(expansion=lambda z: z)
        head.device = torch.device("cpu")
        head.out_features = 0
        head.W = torch.zeros(7, 0, dtype=torch.double)
        head.R = torch.eye(7, dtype=torch.double) / gamma
        end = 0
        for size in chunks:
            start, end = end, end + size
            head.fit(x[start:end], y[start:end])
            targets = F.one_hot(y[:end], head.out_features).double()
            expected = torch.linalg.solve(x[:end].T @ x[:end] + gamma * torch.eye(7, dtype=x.dtype), x[:end].T @ targets)
            torch.testing.assert_close(head.W, expected, rtol=1e-9, atol=1e-10)
        assert end == len(y)
    print("PASS GACL original fit: three batch partitions, repeated/new classes, every prefix vs joint ridge")


def check_air():
    spec = importlib.util.spec_from_file_location("reference_analytic", ROOT / "original/air/analytic/AnalyticLinear.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    x = torch.randn(19, 7, dtype=torch.double)
    y = torch.tensor([0, 0, 0, 1, 1, 0, 1, 0, 0, 2, 2, 1, 2, 0, 2, 1, 1, 0, 2])
    for chunks in ([19], [1] * 19, [3, 6, 2, 8]):
        head = module.GeneralizedARM(7, gamma=0.7, device="cpu")
        end = 0
        for size in chunks:
            start, end = end, end + size
            head.fit(x[start:end], y[start:end])
            head.update()
            counts = torch.bincount(y[:end], minlength=head.out_features).double()
            factors = counts.reciprocal()
            factors *= len(counts) / factors.sum()
            weights = factors[y[:end]]
            a = x[:end].T @ (weights[:, None] * x[:end]) + 0.7 * torch.eye(7, dtype=x.dtype)
            b = x[:end].T @ (weights[:, None] * F.one_hot(y[:end], len(counts)).double())
            expected = torch.linalg.solve(a, b)
            torch.testing.assert_close(head.weight, expected, rtol=1e-9, atol=1e-10)
        assert head.cnt.tolist() == torch.bincount(y).tolist()
    assert "cnt" not in head.state_dict()  # Port must explicitly persist counts.
    print("PASS AIR original ARM: three batch partitions, every prefix vs class-weighted ridge")
    print("AUDIT AIR count tensor is absent from state_dict; adapter checkpoint handling required")


def check_slda():
    official = load_class("original/slda/SLDA_Model.py", "StreamingLDA", cpu_init=True)
    copied = load_class("original/gacl/methods/SLDA.py", "SLDA")
    x = torch.randn(19, 7)
    y = torch.tensor([0, 0, 0, 1, 1, 0, 1, 0, 0, 2, 2, 1, 2, 0, 2, 1, 1, 0, 2])
    head = official(7, 4)
    other = copied.__new__(copied)
    for key in ("device", "input_shape", "shrinkage_param", "streaming_update_sigma", "num_updates"):
        setattr(other, key, getattr(head, key))
    other.muK, other.cK, other.Sigma = head.muK.clone(), head.cK.clone(), head.Sigma.clone()
    for end, (features, label) in enumerate(zip(x, y), 1):
        head.fit(features, label)
        other.fit(features, label)
        for key in ("muK", "cK", "Sigma"):
            torch.testing.assert_close(getattr(head, key), getattr(other, key))
        assert head.num_updates == other.num_updates == end
        for c in y[:end].unique():
            torch.testing.assert_close(head.muK[c], x[:end][y[:end] == c].mean(0))
    scores = head.predict(x)
    covariance = (1 - head.shrinkage_param) * head.Sigma + head.shrinkage_param * torch.eye(7)
    w = torch.linalg.solve(covariance, head.muK.T)
    expected = x @ w - 0.5 * (head.muK.T * w).sum(0)
    torch.testing.assert_close(scores, expected, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(head.predict(x), scores)
    print("PASS SLDA original per-sample updates: means/counts, GACL-copy parity, score formula/cache")

    # Execute original GACL-SLDA online_step with initialization instrumentation.
    # This confirms control flow only; fit_base/OAS is NOT executed by this audit.
    wrapper = copied.__new__(copied)
    wrapper.first_time, wrapper.online_iter = True, 1
    wrapper.device = "cpu"
    wrapper.exposed_classes = [0, 1]
    wrapper.model = SimpleNamespace(features=lambda z: z)
    wrapper.add_new_class = lambda labels: None
    calls = []
    wrapper.fit_base = lambda features, labels: calls.append(len(labels))
    wrapper.predict = lambda features: torch.zeros(len(features), 2)
    wrapper.fit = lambda *args: (_ for _ in ()).throw(AssertionError("Unexpected streaming path"))
    for _ in range(2):
        wrapper.online_step(x[:4], torch.tensor([0, 1, 0, 1]), None)
    assert calls == [4, 4] and wrapper.first_time
    wrapper.online_after_task(0)
    assert wrapper.first_time is False
    print("AUDIT GACL SLDA calls fit_base on each batch until task-end; use original author's streaming core")


if __name__ == "__main__":
    check_sources()
    check_gacl()
    check_air()
    check_slda()
    print(f"PASS reference-core preparation only; torch={torch.__version__}; CPU; no FlyGCL/GPU/accuracy experiment")
