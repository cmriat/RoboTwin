FSDP1 → FSDP2 Migration Guide (Pi repository)

This guide turns the original scratch notes into a clean, actionable Markdown for upgrading from FSDP1 (FullyShardedDataParallel) to FSDP2 (`fully_shard`) in this repository. Where the original draft marked lines with “Pi” as placeholders, this version replaces them with explicit repo references to files and line numbers.

Note on language: repository documentation uses English; code comments remain concise and only in critical paths, per project guidelines.

**Overview**
- Problem with FSDP1: parameter flattening requires uniform dtype within a flattened group. Mixed bf16 + fp32 parameters trigger errors such as “Must flatten tensors with uniform dtype …”.
- Why FSDP2: it shards parameters individually via DTensor rather than flattening large groups, removing the uniform-dtype constraint. Together with `MixedPrecisionPolicy` and Distributed Checkpoint (DCP), this avoids dtype conflicts while preserving your training flow.
- Goal: minimum changes while keeping current “single-file FULL checkpoint on rank0” semantics, then suggest a modern sharded-checkpoint option.

**Current Repo Touchpoints**
- Mixed precision selection: `_select_mixed_precision()` in `scripts/train_pytorch_fsdp.py:241`.
- FSDP1 wrapper: `_fsdp_wrap()` in `scripts/train_pytorch_fsdp.py:254`.
- Checkpoint (FULL, single file): `fsdp_save_full_checkpoint()` at `scripts/train_pytorch_fsdp.py:274` and `fsdp_load_full_checkpoint()` at `scripts/train_pytorch_fsdp.py:320`.
- Preloading non-FSDP weights on rank0: see `scripts/train_pytorch_fsdp.py:432`–`scripts/train_pytorch_fsdp.py:441` (load via `safetensors.torch.load_model`).
- Device move and nested inputs: `_tree_map_to_device()` at `scripts/train_pytorch_fsdp.py:352`; model constructs `Observation` dataclass from a nested dict at `scripts/train_pytorch_fsdp.py:504`.

**Why FSDP2**
- Parameter-level sharding (DTensor) instead of large flattening ⇒ no uniform-dtype requirement, so bf16 + fp32 coexistence is fine.
- `sync_module_states` responsibility moves out: broadcasting from rank0 should be handled via DCP `set_model_state_dict(..., broadcast_from_rank0=True)` semantics instead of FSDP1’s wrap-time sync.
- Mixed precision API changes: FSDP1 `MixedPrecision` → FSDP2 `MixedPrecisionPolicy`. The new policy adds `output_dtype` and `cast_forward_inputs` and removes `buffer_dtype`.
- Checkpointing: prefer DCP (`torch.distributed.checkpoint.state_dict`) to unify full/sharded states and broadcast behavior.

**Minimal Upgrade (preserve current training behavior)**
1) Replace imports and wrapping logic
- Map `FSDP(..., sharding_strategy=FULL_SHARD)` → `fully_shard(module, reshard_after_forward=True)` (ZeRO-3 equivalent).
- Device placement comes from `DeviceMesh` in FSDP2; for a minimal path you can keep your current device logic and let FSDP2 infer shard placement.
- `mixed_precision=MixedPrecision(...)` → `mp_policy=MixedPrecisionPolicy(...)`.
- Remove `sync_module_states=True`; use DCP broadcast on load (see Checkpoint section).

Before (FSDP1)
```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    BackwardPrefetch,
    MixedPrecision,
)

def _select_mixed_precision() -> MixedPrecision | None:
    # Prefer bf16 on Ampere+; keep buffers fp32
    if torch.cuda.is_available():
        sm, _ = torch.cuda.get_device_capability()
        use_bf16 = sm >= 8
        return MixedPrecision(
            param_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            reduce_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            buffer_dtype=torch.float32,
        )
    return None

def _fsdp_wrap(model: torch.nn.Module, device: torch.device) -> FSDP:
    mp = _select_mixed_precision()
    return FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device,
        use_orig_params=True,
        sync_module_states=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        mixed_precision=mp,
        limit_all_gathers=True,
    )
```

After (FSDP2)
```python
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

def _select_mp_policy() -> MixedPrecisionPolicy | None:
    # Prefer bf16 on Ampere+; inputs will be cast at forward
    if torch.cuda.is_available():
        sm, _ = torch.cuda.get_device_capability()
        use_bf16 = sm >= 8
        return MixedPrecisionPolicy(
            param_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            reduce_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            output_dtype=None,
            cast_forward_inputs=True,
        )
    return None

def _fsdp_wrap(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    mp_policy = _select_mp_policy()
    # fully_shard mutates module in place; create optimizer AFTER this
    fully_shard(
        module=model,
        mp_policy=mp_policy,
        reshard_after_forward=True,
    )
    return model
```

2) Checkpoint: keep “FULL single-file on rank0” semantics using DCP
- Replace direct FULL state aggregation calls with DCP state_dict APIs. Use `set_model_state_dict(..., broadcast_from_rank0=True)` to broadcast after rank0 loads.
- This removes the need for `sync_module_states=True` on wrap.

Sketch
```python
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)

# Save (rank0)
model_sd = get_model_state_dict(model)
optim_sd = get_optimizer_state_dict(model, optimizer)
torch.save({"model": model_sd, "optim": optim_sd, "meta": {...}}, path)

# Load (rank0 → all via broadcast)
ckpt = torch.load(path, map_location="cpu")
set_model_state_dict(model, ckpt["model"], broadcast_from_rank0=True)
set_optimizer_state_dict(model, optimizer, ckpt["optim"])
```

3) Forward-input dtype handling for nested inputs
- FSDP2 `MixedPrecisionPolicy(cast_forward_inputs=True)` does not recurse into custom containers like dataclasses. If your model accepts a dataclass (`Observation`) created from nested dicts, ensure inputs are cast appropriately.

Two practical options
- Use autocast around forward: `with torch.autocast(device_type="cuda", dtype=torch.bfloat16): ...`.
- Or update your existing `_tree_map_to_device()` to also normalize float tensors to the parameter dtype (bf16/fp16) as they are moved to device.

**Optional: Modern Sharded Checkpoints**
- For large models, prefer sharded checkpoints (each rank saves its shard). Use `torch.distributed.checkpoint.save/load` with `state_dict.get_*` helpers. This avoids full-model aggregation on any rank and scales better.

**Known Differences and Troubleshooting**
- Dataclass inputs are not auto-cast by FSDP2: use autocast or normalize dtypes in `_tree_map_to_device()`.
- Gradient checkpointing with FSDP2 + autocast: if you hit recompute dtype mismatches, try disabling checkpointing for the affected modules or move autocast into module forward.
- `limit_all_gathers` is not needed with FSDP2.
- `buffer_dtype` no longer exists on `MixedPrecisionPolicy`.

**Version Notes**
- The repo pins PyTorch 2.7.1 in `pixi.toml`. The API references here target the 2.7–2.8 family.

**Repo References (was “Pi” in the draft)**
- FSDP1 wrapper and mixed precision: `scripts/train_pytorch_fsdp.py:241`, `scripts/train_pytorch_fsdp.py:254`.
- FULL checkpoint save/load: `scripts/train_pytorch_fsdp.py:274`, `scripts/train_pytorch_fsdp.py:320`.
- Dataclass input and device move: `scripts/train_pytorch_fsdp.py:352`, `scripts/train_pytorch_fsdp.py:504`.
- Rank0 preload of non-FSDP weights: `scripts/train_pytorch_fsdp.py:432`.

**TL;DR**
- Switch `FSDP → fully_shard` and use `MixedPrecisionPolicy(cast_forward_inputs=True)`.
- Drop `sync_module_states`; broadcast via DCP on load.
- Keep current FULL single-file checkpoint semantics with DCP get/set state-dict helpers.
- Ensure nested inputs are cast (autocast or dtype normalization in `_tree_map_to_device()`).

