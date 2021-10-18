import sys
import os
import pipes
import datetime
import io
import json
import pathlib
import pickle
import re
import time
import uuid

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow_probability as tfp
from tensorflow.keras.mixed_precision import experimental as prec
from tensorflow_probability import distributions as tfd


# Patch to ignore seed to avoid synchronization across GPUs.
_orig_random_categorical = tf.random.categorical
def random_categorical(*args, **kwargs):
  kwargs['seed'] = None
  return _orig_random_categorical(*args, **kwargs)
tf.random.categorical = random_categorical

# Patch to ignore seed to avoid synchronization across GPUs.
_orig_random_normal = tf.random.normal
def random_normal(*args, **kwargs):
  kwargs['seed'] = None
  return _orig_random_normal(*args, **kwargs)
tf.random.normal = random_normal

def save_cmd(base_dir):
  if not isinstance(base_dir, pathlib.Path):
    base_dir = pathlib.Path(base_dir)
  train_cmd = 'python ' + ' '.join([sys.argv[0]] + [pipes.quote(s) for s in sys.argv[1:]])
  train_cmd += '\n'
  print('\n' + '*' * 80)
  print('Training command:\n' + train_cmd)
  print('*' * 80 + '\n')
  with open(base_dir / "cmd.txt", "w") as f:
    f.write(train_cmd)

def save_git(base_dir):
  # save code revision
  print('Save git commit and diff to {}/git.txt'.format(base_dir))
  cmds = ["echo `git rev-parse HEAD` > {}".format(
    os.path.join(base_dir, 'git.txt')),
    "git diff >> {}".format(
      os.path.join(base_dir, 'git.txt'))]
  print(cmds)
  os.system("\n".join(cmds))

class AttrDict(dict):

  __setattr__ = dict.__setitem__
  __getattr__ = dict.__getitem__


class Module(tf.Module):

  def save(self, filename):
    values = tf.nest.map_structure(lambda x: x.numpy(), self.variables)
    amount = len(tf.nest.flatten(values))
    count = int(sum(np.prod(x.shape) for x in tf.nest.flatten(values)))
    print(f'Save checkpoint with {amount} tensors and {count} parameters.')
    with pathlib.Path(filename).open('wb') as f:
      pickle.dump(values, f)

  def load(self, filename):
    with pathlib.Path(filename).open('rb') as f:
      values = pickle.load(f)
    amount = len(tf.nest.flatten(values))
    count = int(sum(np.prod(x.shape) for x in tf.nest.flatten(values)))
    print(f'Load checkpoint with {amount} tensors and {count} parameters.')
    tf.nest.map_structure(lambda x, y: x.assign(y), self.variables, values)

  def get(self, name, ctor, *args, **kwargs):
    # Create or get layer by name to avoid mentioning it in the constructor.
    if not hasattr(self, '_modules'):
      self._modules = {}
    if name not in self._modules:
      self._modules[name] = ctor(*args, **kwargs)
    return self._modules[name]


def var_nest_names(nest):
  if isinstance(nest, dict):
    items = ' '.join(f'{k}:{var_nest_names(v)}' for k, v in nest.items())
    return '{' + items + '}'
  if isinstance(nest, (list, tuple)):
    items = ' '.join(var_nest_names(v) for v in nest)
    return '[' + items + ']'
  if hasattr(nest, 'name') and hasattr(nest, 'shape'):
    return nest.name + str(nest.shape).replace(', ', 'x')
  if hasattr(nest, 'shape'):
    return str(nest.shape).replace(', ', 'x')
  return '?'


class Logger:

  def __init__(self, logdir, step):
    self._logdir = logdir
    self._writer = tf.summary.create_file_writer(str(logdir), max_queue=1000)
    self._last_step = None
    self._last_time = None
    self._scalars = {}
    self._images = {}
    self._videos = {}
    self.step = step

  def scalar(self, name, value):
    self._scalars[name] = float(value)

  def image(self, name, value):
    self._images[name] = np.array(value)

  def video(self, name, value):
    self._videos[name] = np.array(value)

  def write(self, fps=False):
    scalars = list(self._scalars.items())
    if fps:
      scalars.append(('fps', self._compute_fps(self.step)))
    # print(f'[{self.step}]: {self._logdir} , ', ' / '.join(f'{k} {v:.1f}' for k, v in scalars))
    print('step', self.step)
    with (self._logdir / 'metrics.jsonl').open('a') as f:
      f.write(json.dumps({'step': self.step, ** dict(scalars)}) + '\n')
    with self._writer.as_default():
      for name, value in scalars:
        tf.summary.scalar('scalars/' + name, value, self.step)
      for name, value in self._images.items():
        tf.summary.image(name, value, self.step)
      for name, value in self._videos.items():
        video_summary(name, value, self.step)
    self._writer.flush()
    self._scalars = {}
    self._images = {}
    self._videos = {}

  def _compute_fps(self, step):
    if self._last_step is None:
      self._last_time = time.time()
      self._last_step = step
      return 0
    steps = step - self._last_step
    duration = time.time() - self._last_time
    self._last_time += duration
    self._last_step = step
    return steps / duration


def graph_summary(writer, step, fn, *args):
  def inner(*args):
    tf.summary.experimental.set_step(step.numpy().item())
    with writer.as_default():
      fn(*args)
  return tf.numpy_function(inner, args, [])


def video_summary(name, video, step=None, fps=20):
  name = name if isinstance(name, str) else name.decode('utf-8')
  if np.issubdtype(video.dtype, np.floating):
    video = np.clip(255 * video, 0, 255).astype(np.uint8)
  B, T, H, W, C = video.shape
  try:
    frames = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
    summary = tf1.Summary()
    image = tf1.Summary.Image(height=B * H, width=T * W, colorspace=C)
    image.encoded_image_string = encode_gif(frames, fps)
    summary.value.add(tag=name, image=image)
    tf.summary.experimental.write_raw_pb(summary.SerializeToString(), step)
  except (IOError, OSError) as e:
    print('GIF summaries require ffmpeg in $PATH.', e)
    frames = video.transpose((0, 2, 1, 3, 4)).reshape((1, B * H, T * W, C))
    tf.summary.image(name, frames, step)


def encode_gif(frames, fps):
  from subprocess import Popen, PIPE
  h, w, c = frames[0].shape
  pxfmt = {1: 'gray', 3: 'rgb24'}[c]
  cmd = ' '.join([
      f'ffmpeg -y -f rawvideo -vcodec rawvideo',
      f'-r {fps:.02f} -s {w}x{h} -pix_fmt {pxfmt} -i - -filter_complex',
      f'[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse',
      f'-r {fps:.02f} -f gif -'])
  proc = Popen(cmd.split(' '), stdin=PIPE, stdout=PIPE, stderr=PIPE)
  for image in frames:
    proc.stdin.write(image.tostring())
  out, err = proc.communicate()
  if proc.returncode:
    raise IOError('\n'.join([' '.join(cmd), err.decode('utf8')]))
  del proc
  return out

def log_eval_metrics(logger, log_prefix, eval_dir, num_eval_eps):
    #keys = pickle.load(open(str(eval_dir)+'/eval_ep_0.pkl', 'rb')).keys()
    multi_task_data = [pickle.load(open(str(eval_dir)+'/eval_ep_'+str(idx)+'.pkl', 'rb')) for idx in range(num_eval_eps)]
    keys = multi_task_data[0].keys()
    for key in keys:
        for idx in range(num_eval_eps):
            logger.scalar(log_prefix+ 'task_'+str(idx)+'/'+key, multi_task_data[idx][key])

        _avg = np.mean([multi_task_data[idx][key] for idx in range(num_eval_eps)])
        logger.scalar(log_prefix + 'avg/'+ key, _avg)
    logger.write()

def simulate(agent, envs, steps=0, episodes=0, state=None):
  # Initialize or unpack simulation state.
  if state is None:
    step, episode = 0, 0
    done = np.ones(len(envs), np.bool)
    length = np.zeros(len(envs), np.int32)
    obs = [None] * len(envs)
    agent_state = None
  else:
    step, episode, done, length, obs, agent_state = state
  all_rewards = []
  #all_gt_rewards = []
  ep_data_lst= []
  while (steps and step < steps) or (episodes and episode < episodes):
    # Reset envs if necessary.
    if done.any():
      indices = [index for index, d in enumerate(done) if d]
      # promises = [envs[i].reset(blocking=False) for i in indices]
      # for index, promise in zip(indices, promises):
      #   obs[index] = promise()
      results = [envs[i].reset() for i in indices]
      for index, result in zip(indices, results):
        obs[index] = result
  
    # Step agents.
    obs = {k: np.stack([o[k] for o in obs]) for k in obs[0]}
    #action, agent_state = agent(obs, done, agent_state)
    agent_out = agent(obs, done, agent_state)
    if len(agent_out) ==2:
      action, agent_state = agent_out
    else:
      action, agent_state, learned_reward = agent_out
      ep_data = {'learned_reward': learned_reward}
      if 'state' in obs: ep_data['state'] = obs['state']
  
      for key in obs.keys():
        if 'metric_reward' in key:
          ep_data['gt_reward'] = obs[key]
      ep_data_lst.append(ep_data)

    if isinstance(action, dict):
      action = [
          {k: np.array(action[k][i]) for k in action}
          for i in range(len(envs))]
    else:
      action = np.array(action)
    assert len(action) == len(envs)
    # Step envs.
    # promises = [e.step(a, blocking=False) for e, a in zip(envs, action)]
    # obs, _, done = zip(*[p()[:3] for p in promises])
    results = [e.step(a) for e, a in zip(envs, action)]
    obs, _, done = zip(*[p[:3] for p in results])
    obs = list(obs)
    done = np.stack(done)
    episode += int(done.sum())
    length += 1
    step += (done * length).sum()
    length *= (1 - done)
  # Return new state to allow resuming the simulation.
  if len(ep_data_lst) > 0:
    return (step - steps, episode - episodes, done, length, obs, agent_state, ep_data_lst)
  else:
    return (step - steps, episode - episodes, done, length, obs, agent_state)


def save_episodes(directory, episodes):
  directory = pathlib.Path(directory).expanduser()
  directory.mkdir(parents=True, exist_ok=True)
  timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
  filenames = []
  for episode in episodes:
    identifier = str(uuid.uuid4().hex)
    length = len(episode['reward'])
    filename = directory / f'{timestamp}-{identifier}-{length}.npz'
    with io.BytesIO() as f1:
      np.savez_compressed(f1, **episode)
      f1.seek(0)
      with filename.open('wb') as f2:
        f2.write(f1.read())
    filenames.append(filename)
  return filenames


def sample_episodes(episodes, length=None, balance=False, seed=0):
  random = np.random.RandomState(seed)
  while True:
    episode = random.choice(list(episodes.values()))
    if length:
      total = len(next(iter(episode.values())))
      available = total - length
      if available < 1:
        print(f'Skipped short episode of length {total}, need length {length}.')
        continue
      if balance:
        index = min(random.randint(0, total), available)
      else:
        index = int(random.randint(0, available + 1))
      episode = {k: v[index: index + length] for k, v in episode.items()}
    yield episode


def load_episodes(directory, limit=None):
  directory = pathlib.Path(directory).expanduser()
  episodes = {}
  total = 0
  for filename in reversed(sorted(directory.glob('*.npz'))):
    try:
      with filename.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
    except Exception as e:
      print(f'Could not load episode: {e}')
      continue
    episodes[str(filename)] = episode
    total += len(episode['reward']) - 1
    if limit and total >= limit:
      break
  return episodes


class DtypeDist:

  def __init__(self, dist, dtype=None):
    self._dist = dist
    self._dtype = dtype or prec.global_policy().compute_dtype

  @property
  def name(self):
    return 'DtypeDist'

  def __getattr__(self, name):
    return getattr(self._dist, name)

  def mean(self):
    return tf.cast(self._dist.mean(), self._dtype)

  def mode(self):
    return tf.cast(self._dist.mode(), self._dtype)

  def entropy(self):
    return tf.cast(self._dist.entropy(), self._dtype)

  def sample(self, *args, **kwargs):
    return tf.cast(self._dist.sample(*args, **kwargs), self._dtype)


class SampleDist:

  def __init__(self, dist, samples=100):
    self._dist = dist
    self._samples = samples

  @property
  def name(self):
    return 'SampleDist'

  def __getattr__(self, name):
    return getattr(self._dist, name)

  def mean(self):
    samples = self._dist.sample(self._samples)
    return tf.reduce_mean(samples, 0)

  def mode(self):
    sample = self._dist.sample(self._samples)
    logprob = self._dist.log_prob(sample)
    return tf.gather(sample, tf.argmax(logprob))[0]

  def entropy(self):
    sample = self._dist.sample(self._samples)
    logprob = self.log_prob(sample)
    return -tf.reduce_mean(logprob, 0)


class OneHotDist(tfd.OneHotCategorical):

  def __init__(self, logits=None, probs=None, dtype=None):
    self._sample_dtype = dtype or prec.global_policy().compute_dtype
    super().__init__(logits=logits, probs=probs)

  def mode(self):
    return tf.cast(super().mode(), self._sample_dtype)

  def sample(self, sample_shape=(), seed=None):
    # Straight through biased gradient estimator.
    sample = tf.cast(super().sample(sample_shape, seed), self._sample_dtype)
    probs = super().probs_parameter()
    while len(probs.shape) < len(sample.shape):
      probs = probs[None]
    sample += tf.cast(probs - tf.stop_gradient(probs), self._sample_dtype)
    return sample


class GumbleDist(tfd.RelaxedOneHotCategorical):

  def __init__(self, temp, logits=None, probs=None, dtype=None):
    self._sample_dtype = dtype or prec.global_policy().compute_dtype
    self._exact = tfd.OneHotCategorical(logits=logits, probs=probs)
    super().__init__(temp, logits=logits, probs=probs)

  def mode(self):
    return tf.cast(self._exact.mode(), self._sample_dtype)

  def entropy(self):
    return tf.cast(self._exact.entropy(), self._sample_dtype)

  def sample(self, sample_shape=(), seed=None):
    return tf.cast(super().sample(sample_shape, seed), self._sample_dtype)


class UnnormalizedHuber(tfd.Normal):

  def __init__(self, loc, scale, threshold=1, **kwargs):
    self._threshold = tf.cast(threshold, loc.dtype)
    super().__init__(loc, scale, **kwargs)

  def log_prob(self, event):
    return -(tf.math.sqrt(
        (event - self.mean()) ** 2 + self._threshold ** 2) - self._threshold)


class SafeTruncatedNormal(tfd.TruncatedNormal):

  def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
    super().__init__(loc, scale, low, high)
    self._clip = clip
    self._mult = mult

  def sample(self, *args, **kwargs):
    event = super().sample(*args, **kwargs)
    if self._clip:
      clipped = tf.clip_by_value(
          event, self.low + self._clip, self.high - self._clip)
      event = event - tf.stop_gradient(event) + tf.stop_gradient(clipped)
    if self._mult:
      event *= self._mult
    return event


class TanhBijector(tfp.bijectors.Bijector):

  def __init__(self, validate_args=False, name='tanh'):
    super().__init__(
        forward_min_event_ndims=0,
        validate_args=validate_args,
        name=name,
        parameters={})

  def _forward(self, x):
    return tf.nn.tanh(x)

  def _inverse(self, y):
    dtype = y.dtype
    y = tf.cast(y, tf.float32)
    y = tf.where(
        tf.less_equal(tf.abs(y), 1.),
        tf.clip_by_value(y, -0.99999997, 0.99999997), y)
    y = tf.atanh(y)
    y = tf.cast(y, dtype)
    return y

  def _forward_log_det_jacobian(self, x):
    log2 = tf.math.log(tf.constant(2.0, dtype=x.dtype))
    return 2.0 * (log2 - x - tf.nn.softplus(-2.0 * x))


def lambda_return(
    reward, value, pcont, bootstrap, lambda_, axis):
  # Setting lambda=1 gives a discounted Monte Carlo return.
  # Setting lambda=0 gives a fixed 1-step return.
  assert reward.shape.ndims == value.shape.ndims, (reward.shape, value.shape)
  if isinstance(pcont, (int, float)):
    pcont = pcont * tf.ones_like(reward)
  dims = list(range(reward.shape.ndims))
  dims = [axis] + dims[1:axis] + [0] + dims[axis + 1:]
  if axis != 0:
    reward = tf.transpose(reward, dims)
    value = tf.transpose(value, dims)
    pcont = tf.transpose(pcont, dims)
  if bootstrap is None:
    bootstrap = tf.zeros_like(value[-1])
  next_values = tf.concat([value[1:], bootstrap[None]], 0)
  inputs = reward + pcont * next_values * (1 - lambda_)
  returns = static_scan(
      lambda agg, cur: cur[0] + cur[1] * lambda_ * agg,
      (inputs, pcont), bootstrap, reverse=True)
  if axis != 0:
    returns = tf.transpose(returns, dims)
  return returns

class Optimizer(tf.Module):

  def __init__(
      self, name, lr, eps=1e-4, clip=None, wd=None, wd_pattern=r'.*',
      opt='adam'):
    assert 0 <= wd < 1
    assert not clip or 1 <= clip
    self._name = name
    self._clip = clip
    self._wd = wd
    self._wd_pattern = wd_pattern
    self._opt = {
        'adam': lambda: tf.optimizers.Adam(lr, epsilon=eps),
        'nadam': lambda: tf.optimizers.Nadam(lr, epsilon=eps),
        'adamax': lambda: tf.optimizers.Adamax(lr, epsilon=eps),
        'sgd': lambda: tf.optimizers.SGD(lr),
        'momentum': lambda: tf.optimizers.SGD(lr, 0.9),
    }[opt]()
    self._mixed = (prec.global_policy().compute_dtype == tf.float16)
    if self._mixed:
      self._opt = prec.LossScaleOptimizer(self._opt, 'dynamic')

  @property
  def variables(self):
    return self._opt.variables()

  def __call__(self, tape, loss, modules):
    assert loss.dtype in [tf.float32, tf.float16],  self._name
    modules = modules if hasattr(modules, '__len__') else (modules,)
    varibs = tf.nest.flatten([module.variables for module in modules])
    count = sum(np.prod(x.shape) for x in varibs)
    #print(f'Found {count} {self._name} parameters.')
    assert len(loss.shape) == 0, loss.shape
    tf.debugging.check_numerics(loss, self._name + '_loss')
    if self._mixed:
      with tape:
        scaled_loss = self._opt.get_scaled_loss(loss)
    else:
      scaled_loss = loss
    grads = tape.gradient(scaled_loss, varibs)
    if self._mixed:
      grads = self._opt.get_unscaled_gradients(grads)
    norm = tf.linalg.global_norm(grads)
    if not self._mixed:
      tf.debugging.check_numerics(norm, self._name + '_norm')
    if self._clip:
      grads, _ = tf.clip_by_global_norm(grads, self._clip, norm)
    if self._wd:
      self._apply_weight_decay(varibs)
    self._opt.apply_gradients(zip(grads, varibs))
    metrics = {}
    metrics[f'{self._name}_loss'] = loss
    metrics[f'{self._name}_grad_norm'] = norm
    if self._mixed:
      if tf.__version__[:3] == '2.4':
        metrics[f'{self._name}_loss_scale'] = self._opt.loss_scale
      else:
        metrics[f'{self._name}_loss_scale'] = self._opt.loss_scale._current_loss_scale
    return metrics

  def _apply_weight_decay(self, varibs):
    nontrivial = (self._wd_pattern != r'.*')
    if nontrivial:
      print('Applied weight decay to variables:')
    for var in varibs:
      if re.search(self._wd_pattern, self._name + '/' + var.name):
        if nontrivial:
          print('- ' + self._name + '/' + var.name)
        var.assign((1 - self._wd) * var)


def args_type(default):
  def parse_string(x):
    if default is None:
      return x
    if isinstance(default, bool):
      return bool(['False', 'True'].index(x))
    if isinstance(default, int):
      return float(x) if ('e' in x or '.' in x) else int(x)
    if isinstance(default, (list, tuple)):
      return tuple(args_type(default[0])(y) for y in x.split(','))
    return type(default)(x)
  def parse_object(x):
    if isinstance(default, (list, tuple)):
      return tuple(x)
    return x
  return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)


def static_scan(fn, inputs, start, reverse=False):
  last = start
  outputs = [[] for _ in tf.nest.flatten(start)]
  indices = range(len(tf.nest.flatten(inputs)[0]))
  if reverse:
    indices = reversed(indices)
  for index in indices:
    inp = tf.nest.map_structure(lambda x: x[index], inputs)
    last = fn(last, inp)
    [o.append(l) for o, l in zip(outputs, tf.nest.flatten(last))]
  if reverse:
    outputs = [list(reversed(x)) for x in outputs]
  outputs = [tf.stack(x, 0) for x in outputs]
  return tf.nest.pack_sequence_as(start, outputs)


def uniform_mixture(dist, dtype=None):
  if dist.batch_shape[-1] == 1:
    return tfd.BatchReshape(dist, dist.batch_shape[:-1])
  dtype = dtype or prec.global_policy().compute_dtype
  weights = tfd.Categorical(tf.zeros(dist.batch_shape, dtype))
  return tfd.MixtureSameFamily(weights, dist)


def cat_mixture_entropy(dist):
  if isinstance(dist, tfd.MixtureSameFamily):
    probs = dist.components_distribution.probs_parameter()
  else:
    probs = dist.probs_parameter()
  return -tf.reduce_mean(
      tf.reduce_mean(probs, 2) *
      tf.math.log(tf.reduce_mean(probs, 2) + 1e-8), -1)


@tf.function
def cem_planner(
    state, num_actions, horizon, proposals, topk, iterations, imagine,
    objective):
  dtype = prec.global_policy().compute_dtype
  B, P = list(state.values())[0].shape[0], proposals
  H, A = horizon, num_actions
  flat_state = {k: tf.repeat(v, P, 0) for k, v in state.items()}
  mean = tf.zeros((B, H, A), dtype)
  std = tf.ones((B, H, A), dtype)
  for _ in range(iterations):
    proposals = tf.random.normal((B, P, H, A), dtype=dtype)
    proposals = proposals * std[:, None] + mean[:, None]
    proposals = tf.clip_by_value(proposals, -1, 1)
    flat_proposals = tf.reshape(proposals, (B * P, H, A))
    states = imagine(flat_proposals, flat_state)
    scores = objective(states)
    scores = tf.reshape(tf.reduce_sum(scores, -1), (B, P))
    _, indices = tf.math.top_k(scores, topk, sorted=False)
    best = tf.gather(proposals, indices, axis=1, batch_dims=1)
    mean, var = tf.nn.moments(best, 1)
    std = tf.sqrt(var + 1e-6)
  return mean[:, 0, :]


@tf.function
def grad_planner(
    state, num_actions, horizon, proposals, iterations, imagine, objective,
    kl_scale, step_size):
  dtype = prec.global_policy().compute_dtype
  B, P = list(state.values())[0].shape[0], proposals
  H, A = horizon, num_actions
  flat_state = {k: tf.repeat(v, P, 0) for k, v in state.items()}
  mean = tf.zeros((B, H, A), dtype)
  rawstd = 0.54 * tf.ones((B, H, A), dtype)
  for _ in range(iterations):
    proposals = tf.random.normal((B, P, H, A), dtype=dtype)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(mean)
      tape.watch(rawstd)
      std = tf.nn.softplus(rawstd)
      proposals = proposals * std[:, None] + mean[:, None]
      proposals = (
          tf.stop_gradient(tf.clip_by_value(proposals, -1, 1)) +
          proposals - tf.stop_gradient(proposals))
      flat_proposals = tf.reshape(proposals, (B * P, H, A))
      states = imagine(flat_proposals, flat_state)
      scores = objective(states)
      scores = tf.reshape(tf.reduce_sum(scores, -1), (B, P))
      div = tfd.kl_divergence(
          tfd.Normal(mean, std),
          tfd.Normal(tf.zeros_like(mean), tf.ones_like(std)))
      elbo = tf.reduce_sum(scores) - kl_scale * div
      elbo /= tf.cast(tf.reduce_prod(tf.shape(scores)), dtype)
    grad_mean, grad_rawstd = tape.gradient(elbo, [mean, rawstd])
    e, v = tf.nn.moments(grad_mean, [1, 2], keepdims=True)
    grad_mean /= tf.sqrt(e * e + v + 1e-4)
    e, v = tf.nn.moments(grad_rawstd, [1, 2], keepdims=True)
    grad_rawstd /= tf.sqrt(e * e + v + 1e-4)
    mean = tf.clip_by_value(mean + step_size * grad_mean, -1, 1)
    rawstd = rawstd + step_size * grad_rawstd
  return mean[:, 0, :]


class Every:

  def __init__(self, every):
    self._every = every
    self._last = None

  def __call__(self, step):
    if not self._every:
      return False
    if self._last is None:
      self._last = step
      return True
    if step >= self._last + self._every:
      self._last += self._every
      return True
    return False


class EveryNCalls:

  def __init__(self, every):
    self._every = every
    self._last = 0
    self._step = 0
    self.value = False

  def __call__(self, *args):
    if not self._every:
      return False
    self._step += 1
    if self._step >= self._last + self._every or self._last == 0:
      self._last += self._every
      self.value = True
      return True
    self.value = False
    return False


class Once:

  def __init__(self):
    self._once = True

  def __call__(self):
    if self._once:
      self._once = False
      return True
    return False


class Until:

  def __init__(self, until):
    self._until = until

  def __call__(self, step):
    if not self._until:
      return True
    return step < self._until


def schedule(string, step):
  try:
    return float(string)
  except ValueError:
    step = tf.cast(step, tf.float32)
    match = re.match(r'linear\((.+),(.+),(.+)\)', string)
    if match:
      initial, final, duration = [float(group) for group in match.groups()]
      mix = tf.clip_by_value(step / duration, 0, 1)
      return (1 - mix) * initial + mix * final
    match = re.match(r'warmup\((.+),(.+)\)', string)
    if match:
      warmup, value = [float(group) for group in match.groups()]
      scale = tf.clip_by_value(step / warmup, 0, 1)
      return scale * value
    match = re.match(r'exp\((.+),(.+),(.+)\)', string)
    if match:
      initial, final, halflife = [float(group) for group in match.groups()]
      return (initial - final) * 0.5 ** (step / halflife) + final
    raise NotImplementedError(string)

def am_sampling(obs, actions):

  s_t, a_t, s_tp1, r_t = [], [], [], []
  num_trajs, seq_len = obs.shape[:2]
 
  for traj_idx in range(num_trajs):
    end_idx = np.random.randint(1, seq_len)
   
    #positive 
    s_t.append(  tf.concat([obs[traj_idx, end_idx-1], obs[traj_idx, end_idx]], axis =-1))
    s_tp1.append(tf.concat([obs[traj_idx, end_idx],   obs[traj_idx, end_idx]], axis =-1))
    a_t.append(actions[traj_idx, end_idx])
    r_t.append(1)

    #negative
    neg_traj_idx = traj_idx
    while neg_traj_idx == traj_idx:
      neg_traj_idx = np.random.randint(0, num_trajs)
    neg_seq_idx = np.random.randint(1, seq_len)
    s_t.append(  tf.concat([obs[traj_idx, end_idx-1], obs[neg_traj_idx, neg_seq_idx]], axis =-1))
    s_tp1.append(tf.concat([obs[traj_idx, end_idx],   obs[neg_traj_idx, neg_seq_idx]], axis =-1))
    a_t.append(actions[traj_idx, end_idx])
    r_t.append(0)
   
  def _expand_and_concat_tensor(_arr):
    return tf.concat([tf.expand_dims(_elem,axis=0) for _elem in _arr], axis=0)

  s_t =   _expand_and_concat_tensor(s_t)
  s_tp1 = _expand_and_concat_tensor(s_tp1)
  a_t =   _expand_and_concat_tensor(a_t)
  r_t =   tf.convert_to_tensor(r_t, dtype=tf.float16 )
  mask = 1- r_t

  return s_t, a_t, s_tp1, r_t, mask

def get_future_goal_idxs(seq_len, bs):
   
    cur_idx_list = []
    goal_idx_list = []
    #generate indices grid
    for cur_idx in range(seq_len):
      for goal_idx in range(cur_idx, seq_len):
        cur_idx_list.append(np.concatenate([np.ones((bs,1), dtype=np.int32)*cur_idx, np.arange(bs).reshape(-1,1)], axis = -1))
        goal_idx_list.append(np.concatenate([np.ones((bs,1), dtype=np.int32)*goal_idx, np.arange(bs).reshape(-1,1)], axis = -1))
    
    return np.concatenate(cur_idx_list,0), np.concatenate(goal_idx_list,0)

def get_future_goal_idxs_neg_sampling(num_negs, seq_len, bs, batch_len):
    cur_idxs = np.random.randint((0,0), (seq_len, bs), size=(num_negs,2))
    goal_idxs = np.random.randint((0,0), (seq_len, bs), size=(num_negs,2))
    for i in range(num_negs):
      goal_idxs[i,1] = np.random.choice([j for j in range(bs) if j//batch_len != cur_idxs[i,1]//batch_len])
    return cur_idxs, goal_idxs

def get_data_for_off_policy_training(obs, actions, next_obs, goals, relabel_mode='uniform', \
                                      relabel_fraction=0.5, geom_p=0.3, feat_to_embed_func=None):
      
    num_batches, seq_len = obs.shape[:2]
    num_points = num_batches * (seq_len)

    def _reshape(_arr):
      return tf.reshape(_arr, ((num_points,) + tuple(_arr.shape[2:])))
    
    curr_state, action, next_state, _goals = \
        _reshape(obs), _reshape(actions), _reshape(next_obs), _reshape(goals)
    masks = np.ones(num_points, dtype=np.float16) 

    if relabel_fraction > 0:
      # relabelling
      deltas = np.random.geometric(geom_p, size=(num_batches, seq_len))
      next_obs_idxs_for_relabelling = []
      relabel_masks = []
      for i in range(num_batches):
        for j in range(seq_len):
          if relabel_mode == 'uniform':
            next_obs_idxs_for_relabelling.append(np.random.randint((seq_len*i)+j, seq_len * (i + 1) ))

          elif relabel_mode == 'geometric':
            next_obs_idxs_for_relabelling.append(min((seq_len * i) + j + deltas[i, j] - 1, seq_len * (i + 1) - 1))
        
          relabel_masks.append(int(next_obs_idxs_for_relabelling[-1] != ((seq_len*i) + j)))
    
      next_obs_idxs_for_relabelling = np.array(next_obs_idxs_for_relabelling).reshape(-1, 1)
      relabel_masks = np.array(relabel_masks, dtype=np.float16).reshape(-1,1)
    
      idxs = np.random.permutation(num_points).reshape(-1,1)
      relabel_idxs, non_relabel_idxs = np.split(idxs, (int(relabel_fraction*num_points),), axis = 0)
      curr_state, action, next_state = tf.gather_nd(curr_state, idxs), tf.gather_nd(action, idxs),  tf.gather_nd(next_state, idxs)
    
      _relabeled_goals = tf.gather_nd(next_state, next_obs_idxs_for_relabelling[relabel_idxs.flatten()])
      if feat_to_embed_func is not None:
        _relabeled_goals = feat_to_embed_func(_relabeled_goals)

      _goals = tf.concat([ _relabeled_goals, tf.gather_nd(_goals, non_relabel_idxs)], axis = 0)
      masks = tf.concat([ tf.squeeze(relabel_masks[relabel_idxs.flatten()]), tf.gather_nd(masks, non_relabel_idxs)], axis = 0)
  
    s_t = tf.concat([curr_state, _goals], axis=-1)
    s_tp1 = tf.concat([next_state, _goals], axis = -1)
    
    return s_t, action, s_tp1, masks