# Simons training integration

Upstream: <https://github.com/Abel-ai-lab/Simons>, pinned by `PINNED` to
`8e338e915d26c8960713889ef12ab1cf5c08d591`. Upstream is Apache-2.0;
`NOTICE` is retained verbatim. Patches to upstream are distributed under the
same licence. No upstream source checkout, weights or corpus is vendored here.

The complete upstream change set is:

- `0001-restart-streams-and-durable-checkpoints.patch`: retain the original
  role/effect curriculum origin, reseed global/private/per-worker streams for
  a deterministic restart, rebuild banks, add wall-clock saves alongside step
  saves, fsync and atomically publish two alternating snapshot slots with embedded
  content hashes, select by embedded step, keep a stable final checkpoint, and
  expose update measurement hooks. Upstream already used atomic rename; the
  patch strengthens durability and adds selection/retention for guest state.
- `0002-preemptible-job-and-qualification.patch`: add the single-leg runner,
  recipe-preserving width-264 configuration, public univariate/covariates corpus
  validation, embedded state identity and validated stable-final completion handling, append-only restart records, qualification with bounded OOM
  retries and telemetry, and upstream soup/export orchestration.

`prepare.py TREE BUNDLE` verifies the exact upstream HEAD and rejects source or
config drift. It runs `git apply --check` on **all** patches before applying
any of them, in filename order. A second application is an error. The generator
embeds this installer, the pin, notice and patches into each standalone job.

See [stage 2 operations](../../docs/stage2-simons.md) for job generation,
state layout, qualification limits and measured versus unmeasured claims.
