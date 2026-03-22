# Running BareTensor Experiments On Kaggle

This document explains how to run BareTensor experiment scripts on Kaggle and how to choose the accelerator.

## Recommended Accelerator Order

Use this fallback order:

1. TPU `v5e-8`
2. GPU `T4 x2`
3. GPU `P100`

Why:

- TPU is usually the fastest option for these JAX experiments when it actually starts running.
- `T4 x2` is the next best choice for raw training speed.
- `P100` is often the best fallback when the `T4 x2` queue is long.

If a session is still stuck in `QUEUED`, cancel it and move to the next accelerator.

## Install The Kaggle CLI

Install the CLI as a tool, not as a project dependency:

```bash
uv tool install kaggle
```

Why:

- `kaggle` is a developer CLI, not part of the repo runtime.
- This avoids polluting global Python.
- This avoids adding Kaggle to `pyproject.toml` as an app dependency.

## Authentication

Kaggle CLI expects its config under:

```text
~/.kaggle
```

The standard legacy format is:

```json
{
  "username": "YOUR_USERNAME",
  "key": "YOUR_KAGGLE_KEY"
}
```

Save that as:

```text
~/.kaggle/kaggle.json
```

Then:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

If your account uses an access token flow instead, the Kaggle CLI also accepts:

```text
~/.kaggle/access_token
```

with:

```bash
chmod 600 ~/.kaggle/access_token
```

Verify auth:

```bash
kaggle config view
```

You should not see an authentication error.

## Important Kaggle Script Constraints

For script kernels:

- `/kaggle/src` is read-only.
- Write outputs to `/kaggle/working`.
- If your script depends on local helper modules or local datasets, either:
  - inline them into the Kaggle script, or
  - use a Kaggle notebook that clones the repo before running.

For the current BareTensor experiment scripts, the safest script-kernel approach is:

- make the submitted script self-contained,
- download the dataset at runtime if needed,
- write artifacts under `/kaggle/working/artifacts/...`

## Launch Flow

The general pattern is:

1. build a Kaggle kernel bundle
2. push it with the desired accelerator
3. watch the status
4. if it stays queued too long, cancel it
5. retry on the next accelerator

Useful commands:

```bash
kaggle kernels push -p /path/to/kernel_bundle --accelerator TpuV5E8
kaggle kernels push -p /path/to/kernel_bundle --accelerator T4x2
kaggle kernels push -p /path/to/kernel_bundle --accelerator NvidiaTeslaT4
kaggle kernels push -p /path/to/kernel_bundle --accelerator NvidiaTeslaP100

kaggle kernels status <username>/<kernel-slug>
kaggle kernels output <username>/<kernel-slug> -p /tmp/kernel_output
kaggle kernels delete <username>/<kernel-slug> -y
```

Notes:

- In Kaggle UI, `T4 x2` may appear explicitly even if pulled metadata only says `NvidiaTeslaT4`.
- `P100` is usually exposed as `NvidiaTeslaP100`.
- TPU variants may have long queue times.

## Recommended Procedure

### 1. Try TPU first

Submit the kernel with TPU:

```bash
kaggle kernels push -p /path/to/kernel_bundle --accelerator TpuV5E8
```

Poll the status:

```bash
kaggle kernels status <username>/<kernel-slug>
```

If it starts running, keep it.

If it sits in `QUEUED` for too long, cancel it:

```bash
kaggle kernels delete <username>/<kernel-slug> -y
```

Then move to `T4 x2`.

### 2. Fall back to GPU T4 x2

Submit a new kernel slug for the T4 run:

```bash
kaggle kernels push -p /path/to/kernel_bundle_t4 --accelerator T4x2
```

If the UI shows `GPU T4 x2`, keep that run.

If it stays queued too long, cancel it:

```bash
kaggle kernels delete <username>/<kernel-slug> -y
```

Then move to `P100`.

### 3. Fall back to GPU P100

Submit a new kernel slug for the P100 run:

```bash
kaggle kernels push -p /path/to/kernel_bundle_p100 --accelerator NvidiaTeslaP100
```

This is usually the simplest fallback when TPU and T4 queues are bad.

## How To Measure Runtime

Use the script's printed metrics, not just Kaggle page wall time.

Prefer:

- `train_seconds=...`
- `steps_per_second=...`
- `total_seconds=...`

The Kaggle page runtime includes:

- environment startup,
- queue wait effects,
- notebook conversion,
- artifact upload overhead.

So:

- use `train_seconds` for training speed comparisons
- use Kaggle page runtime for end-to-end turnaround comparisons

## Recommended Logging Practice

After a successful run:

1. download the outputs

```bash
kaggle kernels output <username>/<kernel-slug> -p /tmp/kernel_output
```

2. capture:
   - accelerator
   - `train_seconds`
   - `steps_per_second`
   - `total_seconds`
   - train loss
   - validation loss

3. record the result in `docs/learning_log.md`

## Practical Advice

- Keep Kaggle runs self-contained.
- Use a fresh kernel slug when comparing accelerators.
- Do not trust the accelerator until the UI or pulled metadata confirms it.
- If your script only uses one device, extra GPUs will not automatically help.
- Prefer TPU for JAX if it starts promptly.
- Prefer `T4 x2` over `P100` for raw throughput.
- Prefer `P100` over `T4 x2` if queue time is the real bottleneck.
