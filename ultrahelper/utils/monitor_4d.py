import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import List

# --------------------------------------------------------------------------- #
# Utility: register hooks that report every op that touches a 4‑D tensor
# --------------------------------------------------------------------------- #
def _is_4d(t):
    return isinstance(t, torch.Tensor) and t.dim() == 4

def _default_printer(mod: nn.Module, inputs, output):
    in_shapes  = [tuple(t.shape) for t in inputs if isinstance(t, torch.Tensor)]
    out_shapes = [tuple(output.shape)] if isinstance(output, torch.Tensor) \
                 else [tuple(t.shape) for t in output if isinstance(t, torch.Tensor)]
    print(f"[4D] {mod.__class__.__name__}  in{in_shapes}  →  out{out_shapes}")

def register_4d_monitor(model: nn.Module,
                        printer=_default_printer) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Attach a forward‑hook to every leaf module of `model`.
    Whenever a 4‑D tensor appears in inputs or outputs, `printer` is called.

    Returns a list of hook handles so you can remove them later.
    """
    handles = []

    def hook(mod, inp, out):
        if any(_is_4d(t) for t in inp) or _is_4d(out) or \
           (isinstance(out, (list, tuple)) and any(_is_4d(t) for t in out)):
            printer(mod, inp, out)

    for m in model.modules():                    # includes `model` itself
        # skip composite containers so we only report leaf ops (`Conv2d`, etc.)
        if len(list(m.children())) == 0:
            handles.append(m.register_forward_hook(hook))
    return handles


# --------------------------------------------------------------------------- #
# Optional context‑manager wrapper: automatically remove hooks afterwards
# --------------------------------------------------------------------------- #
@contextmanager
def monitor_4d_ops(model: nn.Module, printer=_default_printer):
    handles = register_4d_monitor(model, printer)
    try:
        yield
    finally:                                     # clean up hooks
        for h in handles:
            h.remove()
