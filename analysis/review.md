# Code Review Findings

## 1. Weighted loss ignores the relative weights
The training step multiplies the SmoothL1 loss by the per-sample weights and
then takes the simple arithmetic mean:

```python
loss = (loss_fn(pred, yb) * wb).mean()
```

Because the reduction is still a plain mean, samples with tiny weights are not
actually suppressed relative to samples with large weights—the loss is merely
scaled by the average weight of the batch. To honour the intended weighting, the
weighted sum should be normalised by the sum of the weights (or by the mean
weight) explicitly, e.g. `loss = (loss_fn(pred, yb) * wb).sum() / wb.sum()`.
Otherwise heavily down-weighted samples will continue to influence the gradient
as much as regular samples.【F:analysis/code_snippet.py†L43-L50】

## 2. `prefetch_factor=None` 触发 `TypeError`
当 `NUM_WORKERS` 为 0 时，代码把 `prefetch_factor` 显式设为 `None` 传给
`DataLoader`：

```python
dl_tr = DataLoader(..., num_workers=NUM_WORKERS, prefetch_factor=(4 if NUM_WORKERS > 0 else None))
```

然而 `prefetch_factor` 参数只接受正整数，哪怕工作进程数为 0 也一样；传入
`None` 会在构造 `DataLoader` 时立刻抛出 `TypeError: '>' not supported between
instances of 'NoneType' and 'int'`。建议保留默认值，或仅在 `NUM_WORKERS > 0`
时传递该参数。【F:analysis/code_snippet.py†L26-L41】

## 3. Hard requirement on a second GPU
The script raises a `RuntimeError` whenever the host has fewer than two CUDA
GPUs:

```python
if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise RuntimeError("未检测到第 2 张 GPU。需要 cuda:1。")

torch.cuda.set_device(1)
```

This constraint makes the training script unusable on the far more common
single-GPU (or CPU-only) machines, even though the rest of the code does not
need a dedicated `cuda:1` device. Unless you know the runtime environment will
always provide a second GPU, consider falling back to `cuda:0` or even CPU so
that the script can still run.【F:analysis/code_snippet.py†L53-L59】
