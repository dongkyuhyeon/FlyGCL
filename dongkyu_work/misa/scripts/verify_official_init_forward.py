"""Audit MISA source vs adapter, with identical CPU random ViT-B/16.
No pretrained .npz fetch or real dataset. Does not mutate product model code.
"""
import ast
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, TypeVar
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "dongkyu_work/misa/original/MISA/models/dualprompt.py"
CKPT = ROOT / "dongkyu_work/misa/checkpoints"

def official_classes(vit_factory):
    tree = ast.parse(SRC.read_text())
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and
               node.name in ("Prompt", "DualPrompt")]
    assert [node.name for node in classes] == ["Prompt", "DualPrompt"]
    scope = dict(torch=torch, nn=nn, F=F, Iterable=Iterable,
                 T=TypeVar("T", bound=nn.Module), TypeVar=TypeVar,
                 G_dist=False, g_ratio=False, e_ratio=False,
                 L1Loss=nn.L1Loss(), MSELoss=nn.MSELoss(),
                 OrderedDict=__import__("collections").OrderedDict,
                 timm=SimpleNamespace(create_model=vit_factory),
                 logger=logging.getLogger("official-misa"))
    exec(compile(ast.fix_missing_locations(ast.Module(body=classes,type_ignores=[])),
                 str(SRC),"exec"),scope)
    return scope["Prompt"], scope["DualPrompt"]

def assert_same(tag, left, right, tol=0):
    try:
        torch.testing.assert_close(left, right, rtol=tol, atol=tol)
    except AssertionError as err:
        delta = (left.detach()-right.detach()).abs().max().item()
        raise AssertionError(f"{tag}: max |diff| {delta:.8g}") from err


def verify_independent_backbone_classes():
    """Compare original MISA ViT class with FlyGCL ViT class, no weight copies."""
    import importlib.util
    import warnings
    import gc
    import models.vit as current_vit
    original_path = ROOT / "dongkyu_work/misa/original/MISA/models/vit.py"
    spec = importlib.util.spec_from_file_location("_misa_official_vit_audit", original_path)
    upstream_module = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        spec.loader.exec_module(upstream_module)
    config = dict(img_size=224, patch_size=16, embed_dim=768, depth=12,
                  num_heads=12, num_classes=100, drop_rate=0.0, drop_path_rate=0.0)
    torch.manual_seed(137)
    official_vit = upstream_module.VisionTransformer(**config)
    torch.manual_seed(137)
    local_vit = current_vit.VisionTransformer(**config)
    a, b = official_vit.state_dict(), local_vit.state_dict()
    assert set(a) == set(b), (set(a)^set(b))
    for k in a:
        assert_same("original_ViT_vs_FlyGCL."+k, a[k], b[k])
    assert len(official_vit.blocks) == len(local_vit.blocks) == 12
    print(f"PASS independently constructed ORIGINAL and FlyGCL ViT classes: {len(a)} tensors exact",flush=True)
    del official_vit,local_vit,a,b
    gc.collect()

def main():
    import models.vit as vit
    from dongkyu_work.misa.implementation.model import MISA
    torch.set_num_threads(2)
    verify_independent_backbone_classes()
    seed = 1993
    backbone_constructor = vit.vit_base_patch16_224
    def random_vit(name=None, pretrained=True, num_classes=100):
        assert pretrained is True and num_classes == 100
        return backbone_constructor(pretrained=False,num_classes=num_classes)
    Prompt, Original = official_classes(random_vit)

    def load_prompt(self, filename):
        weight = torch.load(CKPT / Path(filename).name,
                            map_location="cpu",weights_only=True)
        assert weight.shape == self.prompts.shape
        self.prompts = nn.Parameter(weight.clone())

    torch.manual_seed(seed)
    with patch.object(Prompt,"load_prompts_only",load_prompt):
        official = Original(backbone_name="vit_base_patch16_224_l2p",
                            class_num=100,task_num=5,memory_size=0,
                            load_pt=True,isa=False,learnable_mask=False,
                            imbalance=False,e_proj=False,g_proj=False,
                            prompt_func="prompt_tuning",lambd=1.0)
    def integrated_vit(*args,**kw):
        return random_vit(pretrained=kw["pretrained"],num_classes=kw["num_classes"])
    torch.manual_seed(seed)
    with patch.object(vit,"vit_base_patch16_224",integrated_vit):
        integrated = MISA(backbone_name="vit_base_patch16_224",num_classes=100,
                          task_num=5,misa_prompt_dir=str(CKPT),step_num=-1)

    assert len(official.backbone.blocks) == len(integrated.backbone.blocks) == 12
    assert official.backbone.num_features == integrated.backbone.num_features == 768
    assert official.task_id == integrated.task_count == 0
    assert official.num_pt_per_task == integrated.num_pt_per_task == 2
    assert official.len_g_prompt == integrated.len_g_prompt == 10
    assert official.len_e_prompt == integrated.len_e_prompt == 20
    print("PASS architecture/Prompt config (12 ViT blocks, 768 features)",flush=True)

    aa,bb=official.backbone.state_dict(),integrated.backbone.state_dict()
    assert set(aa)==set(bb),(set(aa)^set(bb))
    for key in aa:
        assert_same("seeded backbone."+key,aa[key],bb[key])
    print(f"PASS seeded backbone including FC init: {len(aa)} tensors exact",flush=True)
    for key in ("weight","bias"):
        assert_same("classifier."+key,getattr(official.backbone.fc,key),
                    getattr(integrated.backbone.fc,key))
    for key in ("key","prompts"):
        assert_same("E-Prompt."+key,getattr(official.e_prompt,key),
                    getattr(integrated.e_prompt,key))
    assert official.g_prompt.key is integrated.g_prompt.key is None
    assert_same("G Prompt",official.g_prompt.prompts,integrated.g_prompt.prompts)
    for part,ref in [("g",official.g_prompt),("e",official.e_prompt)]:
        expected=torch.load(CKPT/(part+"_prompt.pt"),map_location="cpu",weights_only=True)
        assert_same("official ISA "+part,ref.prompts,expected)
    print("PASS independent seeded E-Prompt key and FC initialization; official ISA G/E weights",flush=True)

    assert "scale" in dict(official.named_parameters())
    assert "translation" in dict(official.named_parameters())
    assert "scale" not in dict(integrated.named_parameters())
    assert "translation" not in dict(integrated.named_parameters())
    print("DIFF official defines scale/translation; adapter omits them",flush=True)

    torch.manual_seed(7)
    image = torch.randn(2,3,224,224)
    labels=torch.tensor([0,1])
    for training,task in ((False,0),(True,0),(False,1),(True,1)):
        official.task_id=task
        integrated.task_count=task
        official.train(training)
        integrated.train(training)
        official.zero_grad(set_to_none=True)
        integrated.zero_grad(set_to_none=True)
        official.e_prompt.counter.zero_()
        integrated.e_prompt.counter.zero_()
        out_a=official(image)
        out_b=integrated(image)
        assert_same(f"logits task={task} training={training}",out_a,out_b,2e-5)
        assert_same("similarity",official.similarity,integrated.similarity,2e-5)
        assert_same("selection counter",official.e_prompt.counter,
                    integrated.e_prompt.counter)
        if training:
            mask=torch.full((100,),-torch.inf)
            mask[labels]=0
            loss_a=official.loss_fn(out_a+mask,labels)
            loss_b=integrated.loss_fn(out_b+mask,labels)
            assert_same("batch-masked loss",loss_a,loss_b,2e-5)
            loss_a.backward()
            loss_b.backward()
            for name in ("backbone.fc.weight","backbone.fc.bias",
                         "g_prompt.prompts","e_prompt.key","e_prompt.prompts"):
                ga=dict(official.named_parameters())[name].grad
                gb=dict(integrated.named_parameters())[name].grad
                assert ga is not None and gb is not None,name
                assert torch.isfinite(ga).all() and torch.isfinite(gb).all()
                assert_same("gradient."+name,ga,gb,5e-4)
            assert official.scale.grad is None and official.translation.grad is None
            print(f"PASS training task {task}: logits/loss/5 param gradients",flush=True)
        else:
            print(f"PASS evaluation task {task}: raw 100-class logits",flush=True)
    print("PASS official MISA vs FlyGCL independent seeded initialization and full ViT parity",flush=True)
    print("LIMIT same offline-random ViT factory; real upstream .npz loading and GPU/accuracy untested",flush=True)

if __name__ == "__main__":
    main()
