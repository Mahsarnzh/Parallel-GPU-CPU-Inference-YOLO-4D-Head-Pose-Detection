## ultrahelper: Symbolic Traceable C2f Module

This README explains how to reproduce and fix a symbolic tracing issue in the Ultralytics YOLOv8 `C2f` block by providing a `TracableC2f` implementation that can be traced with `torch.fx`.

---

### Table of Contents

1. [Overview](#overview)  
2. [Prerequisites](#prerequisites)  
3. [Reproducing the Tracing Error](#reproducing-the-tracing-error)  
4. [Understanding the Issue](#understanding-the-issue)  
5. [Solution: `TracableC2f`](#solution-tracablec2f)  
6. [Usage](#usage)  
7. [Running Tests](#running-tests)  
8. [Contributing](#contributing)  
9. [License](#license)

---

### Overview

The YOLOv8 `C2f` block uses dynamic control flow based on tensor values, which PyTorch’s FX tracer cannot handle at trace time. We provide a drop‑in replacement, `TracableC2f`, whose `forward()` contains only static operations (splits, concatenations, fixed loops) and is therefore fully FX‑traceable.

---

### Prerequisites

- Python 3.8+  
- PyTorch 1.12+  
- Ultralytics YOLOv8 repository cloned  
- `ultrahelper` installed or symlinked into your project

---

### Reproducing the Tracing Error

```bash
python -m ultrahelper --trace
```

You will see an error like:

```
ValueError: cannot symbolic trace 'if tensor.mean() < 0' in forward of C2f
...
```

That indicates that `C2f.forward()` contains a runtime branch depending on a tensor’s value.

---

### Understanding the Issue

- **FX Tracer** captures tensor operations to build a static graph.  
- **Conditional logic** on tensor values (e.g. `if x.mean() < 0`) cannot be evaluated at trace time.  
- The original `C2f` uses such data‑dependent branching internally, causing `torch.fx` to throw.

---

### Solution: `TracableC2f`

We implement `TracableC2f` without data‑dependent branches. Both the constructor and `forward()` use only fixed‑structure operations:

```python
from ultralytics.nn.modules.block import C2f  # original
from .register import register_module
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import Bottleneck
import torch.nn as nn
import torch

@register_module('base')
@register_module('repeat')
class TracableC2f(nn.Module):
    """
    Symbolic‑traceable version of C2f: no data‑dependent branches.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        # expand and contract channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        # fixed sequence of Bottleneck layers
        self.bottlenecks = nn.Sequential(
            *[Bottleneck(self.c, self.c, shortcut, g,
                         k=((3, 3), (3, 3)), e=1.0)
              for _ in range(n)]
        )

    def forward(self, x):
        # split into two streams
        y1, y2 = self.cv1(x).split((self.c, self.c), 1)
        outputs = [y1, y2]
        # deterministic loop
        for b in self.bottlenecks:
            y2 = b(y2)
            outputs.append(y2)
        # concatenate and project
        return self.cv2(torch.cat(outputs, dim=1))
```

**Key changes**  
- No `if` statements on tensor values.  
- Fixed loop over `bottlenecks`.  
- Uses only `split`, `cat`, and module calls.

---

### Usage

Replace imports of `C2f` with `TracableC2f` in your model config or code:

```python
# before
from ultralytics.nn.modules.block import C2f

# after
from ultrahelper.nn.block import TracableC2f as C2f
```

Then tracing will succeed:

```bash
python -m ultrahelper --trace
# now completes without error
```

---

### Running Tests

1. **Unit tests** for `TracableC2f.forward()` comparing outputs with original `C2f`.  
2. **FX tracing** test:
   ```python
   from torch.fx import symbolic_trace
   tracer = symbol_trace(TracableC2f(64,128,n=3))
   print(tracer.graph)  # should show a clean graph with no Python conditionals
   ```

---

### Contributing

1. Fork the repo & create a feature branch.  
2. Write tests demonstrating the fix.  
3. Submit a PR against `main`.  

---

### License

Released under the [MIT License](LICENSE).

---
*Authored by Mahsa Raeisinezhad.*