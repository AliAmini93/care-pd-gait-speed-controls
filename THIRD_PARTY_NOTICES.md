# Third-party notices

The repository's own code is under the MIT License in [LICENSE](LICENSE). This file covers code that was copied
from other projects.

## Vendored MoMask encoder modules

These four files are inference-only copies of modules from the CARE-PD benchmark repository:

| File here | Source file in TaatiTeam/CARE-PD |
|---|---|
| `scripts/care_pd/vendor_momask_encdec.py` | `model/momask/encdec.py` |
| `scripts/care_pd/vendor_momask_quantizer.py` | `model/momask/quantizer.py` |
| `scripts/care_pd/vendor_momask_residual_vq.py` | `model/momask/residual_vq.py` |
| `scripts/care_pd/vendor_momask_resnet.py` | `model/momask/resnet.py` |

- **Source:** <https://github.com/TaatiTeam/CARE-PD> at commit `9bc22b0258b15369cd8b62f2de20c70b3754349f`, under the
  MIT License, Copyright (c) 2025 TaatiTeam.
- **Upstream origin:** the CARE-PD files derive from MoMask, <https://github.com/EricGuo5513/momask-codes>
  (commit `94a6636c9c463b7a9414c3401a6f1b67e6c51824`), under the MIT License, Copyright (c) 2023 Chuan Guo.
- **Quantizer:** `vendor_momask_quantizer.py` also borrows from vector-quantize-pytorch,
  <https://github.com/lucidrains/vector-quantize-pytorch>, under the MIT License, Copyright (c) 2020 Phil Wang.
- **Changes:** the header comment of each file states what changed. Intra-package import lines were rewritten to the
  flat `vendor_momask_*` module names. In the quantizer, one functional line differs: `reset_codebook` no longer forces
  `.cuda()` on the zero-initialised codebook buffer. `scripts/care_pd/momask_encoder.py` is not a copy. It assembles
  these modules with the option values of the official model configuration.

The full permission text for each copyright holder follows.

### TaatiTeam/CARE-PD

```
MIT License

Copyright (c) 2025 TaatiTeam

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### EricGuo5513/momask-codes

```
MIT License

Copyright (c) 2023 Chuan Guo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### lucidrains/vector-quantize-pytorch

```
MIT License

Copyright (c) 2020 Phil Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## What is not distributed

No checkpoint, normalization file, dataset file or derived array is distributed here. The `fetch-encoder-inputs`
command downloads the pinned MoMask checkpoint and the 16 HumanML3D normalization files from the CARE-PD repository
at the commit above, and the HumanML3D-format inputs from the CARE-PD Dataverse record. It does so only when you pass
`--accept-care-pd-terms`, and the files stay in your own data folder. Those files are covered by their own terms, not
by this repository's licence.
