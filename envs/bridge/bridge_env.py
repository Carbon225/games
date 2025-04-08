import chex
import jax
from .bridge_bidding_2p import BridgeBidding2P, State
from type_aliases import Observation, Reward, Done, Action


env = BridgeBidding2P()
observation_shape = (480,)
num_actions = 38
max_steps = 64


def reset(rng: chex.PRNGKey) -> tuple[State, Observation]:
    state = env.init(rng)
    return state, state.observation


def step(state: State, action: Action) -> tuple[State, Observation, Reward, Done]:
    new_state = env.step(state, action)
    reward = new_state.rewards[state.current_player]
    terminated = new_state.terminated
    truncated = new_state.truncated
    done = terminated | truncated

    observation = new_state.observation

    return new_state, observation, reward, done


def step_autoreset(state: State, action: Action, rng: chex.PRNGKey) -> tuple[State, Observation, Reward, Done]:
    new_state = env.step(state, action)
    reward = new_state.rewards[state.current_player]
    terminated = new_state.terminated
    truncated = new_state.truncated
    done = terminated | truncated

    new_state = jax.lax.cond(
        done,
        lambda: env.init(rng),
        lambda: new_state,
    )

    observation = new_state.observation

    return new_state, observation, reward, done
