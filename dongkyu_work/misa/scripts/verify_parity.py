"""CPU-only parity tests using REAL source definitions from official MISA and FlyGCL.

Uses AST extraction to avoid importing timm/backbone/dataset dependencies and
never runs model training. Run: python dongkyu_work/misa/scripts/verify_parity.py
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]


def source(path):
    return (ROOT / path).read_text(encoding="utf-8")


def extract_class(path, class_name, namespace):
    tree = ast.parse(source(path), filename=path)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    module = ast.Module(body=[cls], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), path, "exec"), namespace)
    return namespace[class_name]


def extract_method(path, class_name, method_name):
    tree = ast.parse(source(path), filename=path)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                  and n.name == method_name)
    namespace = {"torch": torch, "F": F, "nn": nn}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 path, "exec"), namespace)
    return namespace[method_name]


def test_prompt_matches_upstream():
    torch.manual_seed(31415)
    base = {"torch": torch, "F": F, "nn": nn}
    Official = extract_class(
        "dongkyu_work/misa/original/MISA/models/dualprompt.py", "Prompt", dict(base)
    )
    prompt_ns = dict(base)
    Local = extract_class("models/l2p.py", "Prompt", prompt_ns)
    cls_ns = dict(base, Prompt=Local)
    MISAExpertPrompt = extract_class(
        "dongkyu_work/misa/implementation/model.py", "MISAExpertPrompt", cls_ns
    )

    source_prompt = Local(10, 1, 3, 8)
    adapted = MISAExpertPrompt(source_prompt)
    original = Official(10, 1, 3, 8, kwargs={})
    with torch.no_grad():
        original.key.copy_(adapted.key)
        original.prompts.copy_(adapted.prompts)
        # Distinct prompts ensure local vs global gathering differences are observable.
        for i in range(10):
            original.prompts[i].fill_(float(i))
            adapted.prompts[i].fill_(float(i))
    q = torch.randn(4, 8)

    for training in (True, False):
        adapted.train(training)
        original.train(training)
        for span in ((None, None), (0, 2), (2, 4), (4, 6), (8, 10)):
            original.counter.zero_()
            adapted.counter.zero_()
            exp_score, exp_prompt = original(q, s=span[0], e=span[1])
            got_score, got_prompt = adapted(q, s=span[0], e=span[1])
            torch.testing.assert_close(got_score, exp_score, rtol=0, atol=0)
            torch.testing.assert_close(got_prompt, exp_prompt, rtol=0, atol=0)
            torch.testing.assert_close(adapted.counter, original.counter, rtol=0, atol=0)

    # Must differ from the generic FlyGCL Prompt when key slicing starts > 0.
    adapted.train()
    source_prompt.train()
    _, official_selected = adapted(q, s=4, e=6)
    _, flygcl_selected = source_prompt(q, s=4, e=6)
    assert not torch.equal(official_selected, flygcl_selected), (
        "Regression: the original MISA local index was unexpectedly globalized"
    )
    print("PASS prompt parity: released upstream Prompt, 10 train/eval slice cases")


def test_loss_and_training_mask():
    upstream_loss = extract_method(
        "dongkyu_work/misa/original/MISA/models/dualprompt.py", "DualPrompt", "loss_fn"
    )
    fly_loss = extract_method("models/dualprompt.py", "DualPrompt", "loss_fn")
    mock_model = SimpleNamespace(imbalance=False, lambd=1.0,
                                 similarity=torch.tensor(0.37))
    outputs = torch.tensor([[2.0, 0.2, -1.0, 4.0],
                            [0.5, 2.3, 1.2, -0.2]], requires_grad=True)
    labels = torch.tensor([0, 1])
    torch.testing.assert_close(upstream_loss(mock_model, outputs, labels),
                               fly_loss(mock_model, outputs, labels), rtol=0, atol=0)

    orig = extract_method("dongkyu_work/misa/original/MISA/methods/dualprompt.py",
                          "DualPrompt", "model_forward")
    generic = extract_method("methods/dualprompt.py", "DualPrompt", "model_forward")
    fly = extract_method("methods/flyprompt.py", "FlyPrompt", "model_forward")
    batch_mask = torch.full((4,), -torch.inf)
    batch_mask[labels.unique()] = 0
    seen_mask = torch.tensor([0, 0, 0, -torch.inf], dtype=torch.float32)
    dummy = SimpleNamespace(model=lambda x: x.clone(), mask=seen_mask,
                            criterion=nn.CrossEntropyLoss(), use_amp=False)
    for mask in (batch_mask, None):
        expected_logits, expected_loss = orig(dummy, outputs, labels, mask=mask)
        for method in (generic, fly):
            actual_logits, actual_loss = method(dummy, outputs, labels, mask=mask)
            torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
            torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    print("PASS training mask/loss: upstream MISA, FlyGCL DualPrompt and FlyPrompt")


def test_misa_raw_logit_evaluation():
    evaluator = extract_method(
        "dongkyu_work/misa/implementation/trainer.py", "MISA", "online_evaluate"
    )

    class Dummy:
        def eval(self):
            return self

        def __call__(self, x):
            # Unseen classes 2/3 are deliberately strongest: raw-logit
            # MISA accuracy must differ from a seen-class-masked evaluator.
            return torch.tensor([[3.0, 1.0, 9.0, 8.0],
                                 [1.0, 4.0, 3.0, 10.0]], dtype=torch.float32)

    def counts(y, pred):
        return (torch.bincount(y, minlength=4),
                torch.bincount(y[y == pred], minlength=4))

    mock = SimpleNamespace(n_classes=4, model=Dummy(), device="cpu",
                           use_amp=False, topk=1, criterion=nn.CrossEntropyLoss(),
                           exposed_classes=[6, 8], _interpret_pred=counts,
                           mask=torch.tensor([0, 0, -torch.inf, -torch.inf]))
    loader = [(torch.zeros(2, 3), torch.tensor([6, 8]))]
    out = evaluator(mock, loader)
    assert out["avg_acc"] == 0.0, "MISA evaluator must use unmasked raw logits"
    assert set(out.keys()) == {"avg_loss", "avg_acc", "cls_acc"}
    assert len(out["cls_acc"]) == mock.n_classes
    print("PASS MISA evaluation: raw logits, same shared metric schema")


def test_shared_protocol_source():
    root = source("methods/_trainer.py")
    core = source("methods/dualprompt.py")
    fly = source("methods/flyprompt.py")
    model = source("dongkyu_work/misa/implementation/model.py")
    trainer = source("dongkyu_work/misa/implementation/trainer.py")
    runner = source("dongkyu_work/misa/scripts/run_misa.sh")
    registries = source("methods/__init__.py") + source("models/__init__.py")
    for required in (
        "self.train_sampler = OnlineSampler(",
        "OnlineTestSampler(self.test_dataset, self.exposed_classes)",
        "self.online_evaluate(test_dataloader",
        "self.train_sampler.set_task(task_id)",
        "A_auc = np.mean(eval_results",
        "A_last = task_records",
        "self.train_transform = transforms.Compose(",
    ):
        assert required in root, f"Missing shared Trainer protocol: {required}"
    for required in ("class DualPrompt(_Trainer):", "def online_train(", "def model_forward("):
        assert required in core, required
    assert "class FlyPrompt(_Trainer):" in fly
    assert "class MISA(DualPrompt):" in trainer  # inherits common online_train
    assert "class MISA(DualPrompt):" in model
    assert 'self.e_prompt = MISAExpertPrompt(self.e_prompt)' in model
    assert '"misa": MISATrainer' in registries and '"misa": MISA' in registries
    assert "--method misa" in runner and "--rnd_NM" in runner
    assert "--batchsize 64 --online_iter 3" in runner
    assert "--opt_name adam --sched_name default" in runner
    assert '--step_num' in model  # guard against replacing task boundary semantics
    assert "logit + self.mask" in fly  # algorithm-specific eval behavior
    assert "logits = self.model(x)" in trainer and "logits + self.mask" not in trainer
    # For documented CIFAR100 shared protocol: the core sampler is one common import.
    assert "from utils.onlinesampler import OnlineSampler, OnlineTestSampler" in root
    print("PASS protocol wiring: shared sampler, transforms, evaluation cadence, metrics")


if __name__ == "__main__":
    test_prompt_matches_upstream()
    test_loss_and_training_mask()
    test_misa_raw_logit_evaluation()
    test_shared_protocol_source()
    print("PASS: MISA released-algorithm parity and FlyGCL common protocol checks")
