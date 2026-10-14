# Third-party notices

The consistentDMD source retains the parent project's CC BY-NC-SA 4.0 license
in [LICENSE.md](LICENSE.md). Files derived from other projects retain their
original notices and licenses; this release does not relicense those files.

- **FACM / Lightning-DiT:** the Lightning-DiT architecture, VA-VAE implementation,
  and evaluation helpers build on [FACM](https://github.com/ali-vilab/FACM) and
  [Lightning-DiT](https://github.com/hustvl/LightningDiT).
  The copied FACM license is in [licenses/FACM-Apache-2.0.txt](licenses/FACM-Apache-2.0.txt).
  The local model adds consistentDMD's source/target-time conditioning.
- **DiT:** timestep and sinusoidal positional-embedding helpers originate from
  [DiT](https://github.com/facebookresearch/DiT), Copyright Meta Platforms, Inc.
  and affiliates. Their source notices are retained.
- **DMF (Decoupled MeanFlow):** [ldit/flash_attention_2_jvp.py](ldit/flash_attention_2_jvp.py)
  and [ldit/triton_utils.py](ldit/triton_utils.py) adapt the
  [Python FlashAttention-2 JVP implementation](https://github.com/kyungmnlee/dmf/tree/main/models)
  from [DMF](https://github.com/kyungmnlee/dmf). The public adapter retains the
  detached forward-AD path and the split-head Triton kernels for 72-dimensional
  heads. The Triton source also retains its upstream Triton, Ryu1845, and
  Birch-san references. The unmodified [FlashAttention-3 JVP adapter](ldit/flash_attention_3_jvp.py)
  is copied from [DMF's upstream file](https://github.com/kyungmnlee/dmf/blob/a6db0f6135e18f4a5dad6f38df3947ed61960742/models/flash_attention_3_jvp.py).
  No license file is present in the
  [inspected DMF revision](https://github.com/kyungmnlee/dmf/tree/a6db0f6135e18f4a5dad6f38df3947ed61960742).
- **torch-fidelity:** FID and IS use the
  [LTH14 fork](https://github.com/LTH14/torch-fidelity) at commit
  `bfe72b995473e4fee9351705d7b005f17442a4ca`, installed as a dependency.
- **ema-pytorch:** reporting uses `PostHocEMA` from
  [ema-pytorch](https://github.com/lucidrains/ema-pytorch), installed as a dependency.

Model assets are downloaded separately from their upstream repositories and
remain subject to their respective terms. The release's checkpoint is an exact
export of the evaluated consistentDMD reporting average, without the teacher,
optimizer, or historical EMA snapshots.
