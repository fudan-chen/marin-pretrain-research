"""Local initialization candidate, not integrated into Marin.
Call once before a donated step on an EMA-enabled freshly initialized state.
Validated only on the synthetic CPU hierarchy; no GPU/master/offload memory claim.
"""
import dataclasses
import jax
import jax.numpy as jnp
STATUS = 'candidate_not_integrated'
def copy_initial_ema_buffers(state):
    if state.ema_params is None:
        return state
    ema = jax.tree.map(lambda leaf: jnp.array(leaf, copy=True), state.ema_params)
    return dataclasses.replace(state, ema_params=ema)
