import jax
import jax.numpy as jnp

import chex
from chex import PRNGKey

from pgx import State

from type_aliases import Observation, Reward
import envs.bridge.bridge_env as env
import mcts_agent
from modeling.common import NetworkVariables
import az_agent


def evaluate_pvp(rng: PRNGKey, policy1, policy2, batch_size: int):
    def single_move(prev: tuple[State, Observation], rng: PRNGKey) -> tuple[tuple[State, Observation], Reward]:
        state, observation = prev
        rng0, rng1, rng2 = jax.random.split(rng, 3)

        action0 = policy1(rng0, state)
        action1 = policy2(rng1, state)
        action = jnp.where(state.current_player == 0, action0, action1)
    
        new_state, new_observation, new_reward, new_done = jax.vmap(env.step_autoreset)(state, action, jax.random.split(rng2, batch_size))
        return (new_state, new_observation), (-new_reward * (state.current_player * 2 - 1), new_done)

    rng, subkey = jax.random.split(rng)
    state, observation = jax.vmap(env.reset)(jax.random.split(subkey, batch_size))
    
    first = state, observation
    _, out = jax.lax.scan(single_move, first, jax.random.split(rng, env.max_steps))
    rewards, done = out
    chex.assert_shape(rewards, [env.max_steps, batch_size])
    chex.assert_shape(done, [env.max_steps, batch_size])

    num_episodes = done.sum()
    wins = ((rewards[:, :] > 0) & done).sum() / num_episodes
    draws = ((rewards[:, :] == 0) & done).sum() / num_episodes
    losses = ((rewards[:, :] < 0) & done).sum() / num_episodes
    return wins, draws, losses


def random_policy(rng: PRNGKey, state: State) -> chex.Array:
    logits = jnp.zeros(env.num_actions)
    action_mask = state.legal_action_mask
    logits_masked = jnp.where(action_mask, logits, -1e9)
    return jax.random.categorical(rng, logits_masked)


def make_mcts_policy(num_simulations: int):
    def mcts_policy(rng: PRNGKey, state: State) -> chex.Array:
        out = mcts_agent.batched_compute_policy(rng, state, num_simulations)
        logits = out.action_weights
        action_mask = state.legal_action_mask
        logits_masked = jnp.where(action_mask, logits, -1e9)
        return logits_masked.argmax(axis=-1)
    return mcts_policy


def make_az_policy(variables: NetworkVariables, num_simulations: int):
    def az_policy(rng: PRNGKey, state: State) -> chex.Array:
        out = az_agent.batched_compute_policy(variables, rng, state, state.observation, num_simulations)
        logits = out.action_weights
        action_mask = state.legal_action_mask
        logits_masked = jnp.where(action_mask, logits, -1e9)
        return logits_masked.argmax(axis=-1)
    return az_policy


def make_model_policy(variables: NetworkVariables):
    def model_policy(rng: PRNGKey, state: State) -> chex.Array:
        model = variables.merge()
        out = model(state.observation)
        logits = out.pi
        action_mask = state.legal_action_mask
        logits_masked = jnp.where(action_mask, logits, -1e9)
        return logits_masked.argmax(axis=-1)
    return model_policy
