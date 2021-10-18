import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras.mixed_precision import experimental as prec
from tensorflow_probability import distributions as tfd

import networks
import tools
import models
import numpy as np
from tools import get_data_for_off_policy_training

class GCWorldModel(models.WorldModel):
  # TODO can I make this all nicer by only redefining get_init_feat?
  # TODO one way to make it nicer is to redefine the latent state to also contain the goal (with identity transition)

  def __init__(self, step, config):
    super().__init__(step, config)

    if config.pred_stoch_state:
      self.heads['stoch_state'] = networks.DenseHead(
          [self._config.dyn_stoch], config.value_layers, config.units, config.act, std='learned')

    if self._config.gc_reward == 'feat_pca':
      feat_size = self.dynamics._stoch + self.dynamics._deter
      self.feat_roll_mean = tf.Variable(tf.zeros((feat_size,)))
      self.feat_roll_cvar = tf.Variable(tf.zeros((feat_size, feat_size)))

    if self._config.gc_input == 'skills':
      self.rev_pred = networks.get_mlp_model('feat_rp', [256, 128], self._config.skill_dim)
      if self._config.double_rev_pred:
        self.embed_rev_pred = networks.get_mlp_model('embed_rp', [256, 128], self._config.skill_dim)

  def train(self, data):
    data = self.preprocess(data)
    with tf.GradientTape() as model_tape:
      embed = self.encoder(data)
      data['embed'] = tf.stop_gradient(embed)  # Needed for the embed head
      post, prior = self.dynamics.observe(embed, data['action'])
      data['stoch_state'] = tf.stop_gradient(post['stoch'])  # Needed for the embed head
      kl_balance = tools.schedule(self._config.kl_balance, self._step)
      kl_free = tools.schedule(self._config.kl_free, self._step)
      kl_scale = tools.schedule(self._config.kl_scale, self._step)
      kl_loss, kl_value = self.dynamics.kl_loss(
          post, prior, kl_balance, kl_free, kl_scale)
      feat = self.dynamics.get_feat(post)
     
      likes = {}
      for name, head in self.heads.items():
        grad_head = (name in self._config.grad_heads)
        inp = feat
        if name == 'reward':
          inp = tf.concat([feat, self.get_goal(data)], -1)
        if name == 'stoch_state':
          inp = embed
        if not grad_head:
          inp = tf.stop_gradient(inp)
        pred = head(inp, tf.float32)
        like = pred.log_prob(tf.cast(data[name], tf.float32))
        likes[name] = tf.reduce_mean(like) * self._scales.get(name, 1.0)
      model_loss = kl_loss - sum(likes.values())
      if self._config.latent_constraint == 'consecutive_state_l2':
        seq_len = feat.shape[1]
        for i in range(seq_len-1):
          model_loss += 0.1*tf.cast(tf.reduce_mean((feat[:,i,:] - feat[:,i+1,:])**2), tf.float32)
      
      if self._config.gc_input == 'skills' and self._config.double_rev_pred:
        rp_inp = dict(feat=feat, **post)[self._config.skill_pred_input]
        skill_target = tf.stop_gradient(self.rev_pred(rp_inp))
        skill_loss = tf.reduce_mean((self.embed_rev_pred(embed) - skill_target) ** 2)
        model_loss += tf.cast(skill_loss, tf.float32)

    model_parts = [self.encoder, self.dynamics] + list(self.heads.values())
    if self._config.gc_input == 'skills' and self._config.double_rev_pred:
        model_parts += [self.embed_rev_pred]

    metrics = self._model_opt(model_tape, model_loss, model_parts)
    metrics.update({f'{name}_loss': -like for name, like in likes.items()})
    metrics['kl_balance'] = kl_balance
    metrics['kl_free'] = kl_free
    metrics['kl_scale'] = kl_scale
    metrics['kl'] = tf.reduce_mean(kl_value)
    metrics['prior_ent'] = self.dynamics.get_dist(prior).entropy()
    metrics['post_ent'] = self.dynamics.get_dist(post).entropy()
    return embed, post, feat, kl_value, metrics

  def get_goal(self, obs, training=False):
    if self._config.gc_input == 'state':
      assert self._config.training_goals == 'env'
      return obs['goal']
    else:
      if (not training) or self._config.training_goals == 'env':
        # Never alter the goal when evaluating
        _embed = self.encoder({'image': obs['image_goal'], 'state': obs.get('goal', None)})
        if self._config.gc_input == 'embed':
          return _embed
        elif 'feat' in self._config.gc_input:
          return self.get_init_feat_embed(_embed) if len(_embed.shape) == 2 else tf.vectorized_map(self.get_init_feat_embed, _embed)
        elif self._config.gc_input == 'skills':
          if 'skill' in obs and tf.size(obs['skill']) > 0:
            return obs['skill']
          if self._config.double_rev_pred:
            return self.embed_rev_pred(_embed)
          if self._config.skill_pred_input == 'embed':
            return self.rev_pred(_embed)
          raise NotImplementedError

      elif self._config.training_goals == 'batch' and self._config.gc_input == 'skills':
        sh = obs['image_goal'].shape
        if len(sh) == 4:
          return tf.random.uniform((sh[0], self._config.skill_dim), -1, 1, dtype=self._float)
        elif len(sh) == 5:
          return tf.random.uniform((sh[0], sh[1], self._config.skill_dim), -1, 1, dtype=self._float)

      elif self._config.training_goals == 'batch':
        # Use random goals from the same batch
        # This is only run during imagination training
        goal_embed = self.encoder(obs)
        sh = goal_embed.shape
        goal_embed = tf.reshape(goal_embed, (-1, sh[-1]))
        # goal_embed = tf.random.shuffle(goal_embed)  # shuffle doesn't have gradients so need this workaround...
        ids = tf.random.shuffle(tf.range(tf.shape(goal_embed)[0]))
        if self._config.labelled_env_multiplexing:
          # Shufle from the same envs for the multi-env agent
          l = tf.shape(goal_embed)[0]
          env_ids = tf.reshape(obs['env_idx'], [-1])
          oh_ids = tf.one_hot(env_ids, l)  # rows are samples, columns are envs
          cooc = tf.gather(oh_ids, env_ids, axis=1)  # co-occurence matrix, mat_ij = 0 <==> i,j are the same env
          # Sample from the same env
          ids = tf.random.categorical(tf.math.log(cooc / tf.reduce_sum(cooc, 1, keepdims=True)), 1)[:, 0]
  
        goal_embed = tf.gather(goal_embed, ids)
        goal_embed = tf.reshape(goal_embed, sh)
  
        if 'feat' in self._config.gc_input:
          return tf.vectorized_map(self.get_init_feat_embed, goal_embed)
        else:
          return goal_embed
        
  def get_init_feat(self, obs, state=None, sample=False):
    # For GCDreamer, the state of the agent contains the goal
    if state is None:
      batch_size = len(obs['image'])
      latent = self.dynamics.initial(len(obs['image']))
      dtype = prec.global_policy().compute_dtype
      action = tf.zeros((batch_size, self._config.num_actions), dtype)
      goal = obs['image_goal']
      goal_state = obs['goal']
      skill = obs['skill']
    else:
      latent, action = state
      goal = latent['image_goal']
      goal_state = latent['goal']
      skill = latent['skill']
    embed = self.encoder(obs)
    latent, _ = self.dynamics.obs_step(latent, action, embed, sample)
    # TODO encapsulate the goal into obs_step
    latent['image_goal'] = goal
    latent['goal'] = goal_state
    latent['skill'] = skill
    feat = self.dynamics.get_feat(latent)
    return feat, latent
