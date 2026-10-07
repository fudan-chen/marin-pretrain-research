"""Local diagnostic candidate; no integration, failure policy or distributed consensus.
Caller supplies stored bias and the result of original apply_qb_betas.
Finite flags do not prove route/update correctness or undo an earlier divergent update.
"""
import jax.numpy as jnp
STATUS = 'candidate_not_integrated'
def qb_health_flags(pending_beta, stored_bias, next_forward_bias):
    return jnp.stack([
        jnp.all(jnp.isfinite(pending_beta)),
        jnp.all(jnp.isfinite(stored_bias)),
        jnp.all(jnp.isfinite(next_forward_bias)),
    ])
