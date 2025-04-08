import jax
import jax.numpy as jnp
from pgx import State
from chex import PRNGKey
import mctx

from type_aliases import Action, Observation
import envs.bridge.bridge_env as env
from modeling.common import NetworkVariables


def act_randomly(rng: PRNGKey, state: State) -> Action:
    logits = jnp.log(state.legal_action_mask.astype(jnp.float16))
    return jax.random.categorical(rng, logits)


def batched_root_fn(variables: NetworkVariables, rng: PRNGKey, state: State, observation: Observation) -> mctx.RootFnOutput:
    model = variables.merge()
    outputs = model(observation)
    return mctx.RootFnOutput(
        prior_logits=outputs.pi,
        value=outputs.v,
        embedding=state,
    )


def batched_recurrent_fn(variables: NetworkVariables, rng: PRNGKey, action: Action, state: State):
    model = variables.merge()
    new_state, observation, reward, done = jax.vmap(env.step)(state, action)
    outputs = model(observation)
    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=jnp.where(done, 0.0, -1.0).astype(jnp.float32),
        prior_logits=outputs.pi,
        value=jnp.where(done, 0.0, outputs.v).astype(jnp.float32),
    )
    return recurrent_fn_output, new_state


def batched_compute_policy(variables: NetworkVariables, rng: PRNGKey, state: State, observation: Observation, num_simulations: int) -> mctx.PolicyOutput:
    policy_rng, root_rng = jax.random.split(rng, 2)
    policy_output = mctx.gumbel_muzero_policy(
        params=variables,
        rng_key=policy_rng,
        root=batched_root_fn(variables, root_rng, state, observation),
        recurrent_fn=batched_recurrent_fn,
        num_simulations=num_simulations,
        max_depth=env.max_steps,
        qtransform=mctx.qtransform_completed_by_mix_value,
        invalid_actions=1.0 - state.legal_action_mask.astype(jnp.float32),
        gumbel_scale=1.0,
    )
    return policy_output
