from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import optax
from einops import rearrange
import pandas as pd
from tqdm import tqdm
import wandb

from jax.config import config
config.update("jax_numpy_rank_promotion", "raise")

from wordle_utils import score_guess


def build_functions():
    def encoder(guess, score):
      letter_emb = hk.Embed(26, 112, name='letter_emb')(guess)
      guess_emb  = hk.Embed(26,  16, name='guess_emb')(score)
      x = jnp.concatenate([letter_emb, guess_emb], axis=-1)
      x = rearrange(x, 'batch letter dim -> batch (letter dim)')
      return hk.Sequential([
        hk.Linear(1024, with_bias=False, name='encoder_fc1'),
        hk.LayerNorm(-1, True, True, name='encoder_ln1'),
        jax.nn.relu,
        hk.Linear(1024, with_bias=False, name='encoder_fc2'),
        hk.LayerNorm(-1, True, True, name='encoder_ln2'),
        jax.nn.relu,
        hk.Linear(1024, with_bias=False, name='encoder_fc3'),
      ])(x)

    def actor(current_information):
      global guesses
      features = hk.Sequential([
        hk.Linear(1024, with_bias=False, name='actor_fc1'),
        hk.LayerNorm(-1, True, True, name='actor_ln1'),
        jax.nn.relu,
        hk.Linear(1024, with_bias=False, name='actor_fc2'),
        hk.LayerNorm(-1, True, True, name='actor_ln2'),
        jax.nn.relu,
      ])(current_information)
      return hk.Linear(len(guesses))(features)

    def critic(current_information):
      global guesses
      features = hk.Sequential([
        hk.Linear(1024, with_bias=False, name='critic_fc1'),
        hk.LayerNorm(-1, True, True, name='critic_ln1'),
        jax.nn.relu,
        hk.Linear(1024, with_bias=False, name='critic_fc2'),
        hk.LayerNorm(-1, True, True, name='critic_ln2'),
        jax.nn.relu,
      ])(current_information)
      return hk.Linear(len(guesses), w_init=jnp.zeros, b_init=jnp.zeros)(features)

    def init(guess, score):
      info = encoder(guess, score)
      actions = actor(info)
      critics = critic(info)
      return actions, critics

    return init, (encoder, actor, critic)


def unpack_without_apply_rng(multitransformed):
  funs = multitransformed.apply
  def apply_without_rng(fun):
    def inner(params, *args, **kwargs):
      return fun(params, None, *args, **kwargs)
    return inner
  return jax.tree_map(apply_without_rng, funs)

@partial(jax.vmap, in_axes=[-1, None], out_axes=-1)
@partial(jax.vmap, in_axes=[None, 0], out_axes=0)
def select_guess(guesses, idx):
    return guesses[idx]

@jax.jit
def training_step(key, solutions, opt_state, params):
  def calculate_loss(θ):
    B = solutions.shape[0]
    information = jnp.zeros([B, 1024])

    evaluations = []
    scores = []
    expected_rewards = []

    for _ in range(6):
      guess_logits = actor(θ, information)
      evaluation = critic(θ, information)
      guess_idx = jax.random.categorical(key, guess_logits)
      guess = select_guess(guesses, guess_idx)
      evaluations.append(
        jnp.take_along_axis(evaluation, guess_idx.reshape(-1, 1), axis=-1)[..., 0])

      expected_rewards.append(jnp.mean(jax.nn.log_softmax(guess_logits, axis=-1) *
                              jax.lax.stop_gradient(evaluation), axis=-1))

      score = jax.vmap(score_guess)(guess, solutions)
      scores.append(score)
      information = information + encoder(θ, guess, score)

    scores = jnp.stack(scores, axis=1)
    solved = jnp.all(scores == 2, axis=-1)
    reward = solved.sum(axis=1)
    critic_loss = sum(jnp.mean(jnp.square(ev - reward)) for ev in evaluations)
    expected_reward = jnp.mean(jnp.stack(expected_rewards))
    actor_loss = -expected_reward

    return critic_loss + actor_loss, dict(
        critic_loss=critic_loss,
        actor_loss=actor_loss,
        expected_reward=expected_reward,
        actual_reward=jnp.mean(reward),
        pct_solved=jnp.mean(solved),
    )

  grads, metrics = jax.grad(calculate_loss, has_aux=True)(params)
  updates, opt_state = opt.update(grads, opt_state)
  params = optax.apply_updates(params, updates)
  return metrics, opt_state, params


def train_epoch(key, opt_state, params):
  metrics = {}
  for subepoch in range(10):
    key, subkey = jax.random.split(key)
    batch_data = jax.random.permutation(key, guesses, axis=-1)
    N = batch_data.shape[0]
    batch_data = batch_data[:(N//batch_size)*batch_size]
    for batch in rearrange(batch_data, '(N B) ... -> N B ...', B=batch_size):
      step_metrics, opt_state, params = training_step(key, batch, opt_state, params)
      if not metrics: metrics = {k: [] for k in step_metrics}
      metrics = jax.tree_multimap(lambda x, acc: acc + [x], step_metrics, metrics)
  return metrics, opt_state, params


if __name__ == '__main__':
  data = np.load('lists.npz')
  solutions = data['solutions']
  guesses = data['guesses']

  net = hk.multi_transform(build_functions)
  rng = jax.random.PRNGKey(42)

  guess = jnp.ones([7, 5], dtype=jnp.uint8)
  score = jnp.ones([7, 5], dtype=jnp.uint8)
  params = net.init(rng, guess, score)
  encoder, actor, critic = unpack_without_apply_rng(net)
  opt = optax.adam(1e-3, b1=0.5, b2=0.9)
  opt_state = opt.init(params)

  batch_size = 1024

  wandb.init(project='WordleRL')
  key = jax.random.PRNGKey(0)
  for epoch in tqdm(range(2000)):
    key, subkey = jax.random.split(key)
    metrics, opt_state, params = train_epoch(subkey, opt_state, params)
    metrics = {k: np.mean(v) for k, v in metrics.items()}
    print(metrics)
    wandb.log(metrics, step=epoch)
