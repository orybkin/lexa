import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras.mixed_precision import experimental as prec
from tensorflow_probability import distributions as tfd

import networks
import tools
import models
import numpy as np
from tools import get_data_for_off_policy_training, get_future_goal_idxs, get_future_goal_idxs_neg_sampling

def get_mlp_model(name, hidden_units, out_dim):
  with tf.name_scope(name) as scope:
    model = tfk.Sequential()
    for units in hidden_units:
        model.add(tfk.layers.Dense(units, activation='elu'))
    model.add(tfk.layers.Dense(out_dim, activation='tanh'))
  return model

def assign_cond(x, cond, y):
  cond = tf.cast(cond, x.dtype)
  return x * (1 - cond) + y * cond

class GCDreamerBehavior(models.ImagBehavior):

  def __init__(self, config, world_model, stop_grad):

    super(GCDreamerBehavior, self).__init__(config, world_model, stop_grad)
    kw = dict(wd=config.weight_decay, opt=config.opt)
    if self._config.gc_input == 'skills':
      self._rp_opt = tools.Optimizer(
        'rev_pred', config.rp_lr, config.opt_eps, config.rp_grad_clip, **kw)
    if self._config.gc_reward == 'dynamical_distance':
      assert self._config.dd_distance in ['steps_to_go', 'binary']
      if self._config.dd_loss == 'regression':
        dd_out_dim = 1 
        self.dd_loss_fn = tf.keras.losses.MSE
      else:
        if self._config.dd_train_off_policy and self._config.dd_train_imag:
          raise NotImplementedError
        dd_out_dim = self._config.imag_horizon if self._config.dd_distance == 'steps_to_go' else 2
        if self._config.dd_distance == 'binary':
          dd_out_dim = 2
        elif self._config.dd_distance == 'steps_to_go':
          dd_out_dim = self._config.imag_horizon
          if self._config.dd_neg_sampling_factor>0:
            dd_out_dim +=1
        self.dd_loss_fn = tf.keras.losses.CategoricalCrossentropy()

      if self._config.dd_train_off_policy and self._config.dd_train_imag:
        self.dd_seq_len = max(self._config.batch_length, self._config.imag_horizon)
      elif self._config.dd_train_imag:
        self.dd_seq_len = self._config.imag_horizon
      else:
        self.dd_seq_len = self._config.batch_length

      self.dd_out_dim = dd_out_dim
      self.dynamical_distance = networks.GC_Distance(out_dim = dd_out_dim, 
                                    input_type= self._config.dd_inp, normalize_input = self._config.dd_norm_inp)
      self.dd_cur_idxs, self.dd_goal_idxs = get_future_goal_idxs(seq_len = self._config.imag_horizon, 
                                    bs = self._config.batch_size*self._config.batch_length)
      self._dd_opt = tools.Optimizer(
            'dynamical_distance_opt', config.value_lr, config.opt_eps, config.value_grad_clip, **kw)

  def get_actor_inp(self, feat, goal, repeats=None):
    # Image and goal together - input to the actor
    goal = tf.reshape(goal, [1, feat.shape[1], -1])
    goal = tf.repeat(goal, feat.shape[0], 0)
    if repeats:
      goal = tf.repeat(tf.expand_dims(goal, 2), repeats,2)

    return tf.concat([feat, goal], -1)

  def act(self, feat, obs, latent):

    goal = self._world_model.get_goal(latent)
    _state_rep_dict = {'feat': feat, 'embed': self._world_model.encoder(self._world_model.preprocess(obs))}  
    state = _state_rep_dict[self._config.state_rep_for_policy]
    return self.actor(state, goal)
  
  def train_dd_off_policy(self, off_pol_obs):
    obs = tf.transpose(off_pol_obs, (1,0,2))
    with tf.GradientTape() as df_tape:
      dd_loss = self.get_dynamical_distance_loss(obs, corr_factor = 1)
    return self._dd_opt(df_tape, dd_loss, [self.dynamical_distance])

  def _gc_reward(self, feat, inp_state=None, action=None, obs=None):
    
    #image embedding as goal
    if self._config.gc_input == 'embed':
      inp_feat, goal_embed = tf.split(feat, [-1, self._world_model.embed_size], -1)

      if self._config.gc_reward == 'l2':
        goal_feat = tf.vectorized_map(self._world_model.get_init_feat_embed, goal_embed)
        return -tf.reduce_mean((goal_feat - inp_feat) ** 2, -1)
      
      elif self._config.gc_reward == 'cosine':
        goal_feat = tf.vectorized_map(self._world_model.get_init_feat_embed, goal_embed)
        norm = tf.norm(goal_feat, axis =-1)*tf.norm(inp_feat, axis = -1)
        dot_prod = tf.expand_dims(goal_feat,2)@tf.expand_dims(inp_feat,3)
        return tf.squeeze(dot_prod)/(norm+1e-8)

      elif self._config.gc_reward == 'dynamical_distance':
        if self._config.dd_inp == 'feat':
          inp_feat = inp_state['stoch']
          goal_feat = tf.vectorized_map(self._world_model.get_init_state_embed, goal_embed)['stoch']
          if len(inp_feat.shape)==2:
            inp_feat = tf.expand_dims(inp_feat,0)
          dd_out = self.dynamical_distance(tf.concat([inp_feat, goal_feat], axis =-1))

        elif self._config.dd_inp == 'embed': 
          inp_embed = self._world_model.heads['embed'](inp_feat).mode()
          dd_out = self.dynamical_distance(tf.concat([inp_embed, goal_embed], axis =-1))

        if self._config.dd_loss == 'regression':
          reward = -dd_out 
        else:
          reward = - tf.squeeze(tf.math.reduce_sum(dd_out*np.arange(self.dd_out_dim), axis = -1))  
          if self._config.dd_distance == 'steps_to_go':
            reward/= self.dd_seq_len
        return reward

    #latent as goal   
    elif 'feat' in self._config.gc_input:
      inp_feat , goal_feat = tf.split(feat, 2, -1)
      if self._config.gc_reward == 'l2':
        return -tf.reduce_mean((goal_feat - inp_feat) ** 2, -1)

      if self._config.gc_reward == 'cosine':
        norm = tf.norm(goal_feat, axis =-1)*tf.norm(inp_feat, axis = -1)
        dot_prod = tf.expand_dims(goal_feat,2)@tf.expand_dims(inp_feat,3)
        return tf.squeeze(dot_prod)/(norm+1e-8)
      
      elif self._config.gc_reward == 'dynamical_distance':
        raise AssertionError('should use embed as gc_input')

    #skill space goal
    elif self._config.gc_input == 'skills':
      if not inp_state:
        raise NotImplementedError
      inp_feat, skill = tf.split(feat, [-1, self._config.skill_dim], -1)
      if self._config.skill_pred_input == 'embed':
        rp_inp = inp_embed = self._world_model.heads['embed'](inp_feat).mode()
      else:
        rp_inp = dict(feat=inp_feat, **inp_state)[self._config.skill_pred_input]
      if self._config.skill_pred_noise > 0:
        noise = tf.random.normal(rp_inp.shape, dtype=self._world_model._float) * self._config.skill_pred_noise
        rp_inp = rp_inp + noise

      pred_skill = self._world_model.rev_pred(rp_inp)
      loss = (pred_skill - skill)**2
      return -tf.reduce_mean(loss, -1)
  
    if self._config.gc_reward == 'env':
      return self._world_model.heads['reward'](feat).mode()
 
  def get_dynamical_distance_loss(self, _data, corr_factor = None):
   
    seq_len, bs = _data.shape[:2]
  
    def _helper(cur_idxs, goal_idxs, distance):
      loss = 0
      cur_states = tf.expand_dims(tf.gather_nd(_data, cur_idxs),0)
      goal_states = tf.expand_dims(tf.gather_nd(_data, goal_idxs),0)
      pred = tf.cast(self.dynamical_distance(tf.concat([cur_states, goal_states], axis=-1)), tf.float32)
     
      if self._config.dd_loss == 'regression':
        _label = distance
        if self._config.dd_norm_reg_label and self._config.dd_distance == 'steps_to_go':
          _label = _label/self.dd_seq_len
        loss += tf.reduce_mean((_label-pred)**2)
      else:
        _label = tf.one_hot(tf.cast(distance, tf.int32), self.dd_out_dim)
        loss += self.dd_loss_fn(_label, pred)
      return loss
    
    #positives
    idxs = np.random.choice(np.arange(len(self.dd_cur_idxs)), self._config.dd_num_positives)
    loss = _helper(self.dd_cur_idxs[idxs], self.dd_goal_idxs[idxs], self.dd_goal_idxs[idxs][:,0] - self.dd_cur_idxs[idxs][:,0])

    #negatives
    corr_factor = corr_factor if corr_factor != None else self._config.batch_length
    if self._config.dd_neg_sampling_factor>0:
      num_negs = int(self._config.dd_neg_sampling_factor*self._config.dd_num_positives)
      neg_cur_idxs, neg_goal_idxs = get_future_goal_idxs_neg_sampling(num_negs, seq_len, bs, corr_factor)
      loss += _helper(neg_cur_idxs, neg_goal_idxs, tf.ones(num_negs)*seq_len)

    return loss

  def train(
      self, start, imagine=None, tape=None, repeats=None, obs=None):

    self._update_slow_target()
    metrics = {}

    with (tape or tf.GradientTape(persistent=self._config.gc_input == 'skills')) as actor_tape:
      obs = self._world_model.preprocess(obs)
      goal = self._world_model.get_goal(obs, training=True)

      imag_feat, imag_state, imag_action = self._imagine(
        start, self.actor, self._config.imag_horizon, repeats, goal)
      actor_inp = self.get_actor_inp(imag_feat, goal)
      reward = self._gc_reward(actor_inp, imag_state, imag_action, obs)
      if self._config.gc_input == 'skills':
        rp_loss = tf.cast(-tf.reduce_mean(reward), tf.float32)

      actor_ent = self.actor(actor_inp, dtype=tf.float32).entropy()
      state_ent = self._world_model.dynamics.get_dist(
        imag_state, tf.float32).entropy()

      target, weights = self._compute_target(
          actor_inp, reward, actor_ent, state_ent,
          self._config.slow_actor_target)

      actor_loss, mets = self._compute_actor_loss(
            actor_inp, imag_state, imag_action, target, actor_ent, state_ent,
            weights)
      metrics.update(mets)

    if self._config.slow_value_target != self._config.slow_actor_target:
      target, weights = self._compute_target(
          actor_inp, reward, actor_ent, state_ent,
          self._config.slow_value_target)
    
    metrics['reward_mean'] = tf.reduce_mean(reward)
    metrics['reward_std'] = tf.math.reduce_std(reward)
    metrics['actor_ent'] = tf.reduce_mean(actor_ent)
    metrics.update(self._actor_opt(actor_tape, actor_loss, [self.actor]))
     
    if self._config.gc_input == 'skills':
      metrics.update(self._rp_opt(actor_tape, rp_loss, [self._world_model.rev_pred]))
      del actor_tape

    if self._config.gc_reward == 'dynamical_distance' and self._config.dd_train_imag:
      with tf.GradientTape() as df_tape:
        _inp = imag_state['stoch'] if 'feat' in self._config.dd_inp \
              else self._world_model.heads['embed'](imag_feat).mode()
        dd_loss = self.get_dynamical_distance_loss(_inp)
      metrics.update(self._dd_opt(df_tape, dd_loss, [self.dynamical_distance]))
   
    with tf.GradientTape() as value_tape:
      value = self.value(actor_inp, tf.float32)[:-1]
      value_loss = -value.log_prob(tf.stop_gradient(target))
      if self._config.value_decay:
        value_loss += self._config.value_decay * value.mode()
      value_loss = tf.reduce_mean(weights[:-1] * value_loss)
    metrics.update(self._value_opt(value_tape, value_loss, [self.value]))

    return imag_feat, imag_state, imag_action, weights, metrics

  def _imagine(self, start, policy, horizon, repeats=None, goal=None):
    dynamics = self._world_model.dynamics
    flatten = lambda x: tf.reshape(x, [-1] + list(x.shape[2:]))
    goal = flatten(goal)
    if repeats:
      start = {k: tf.repeat(v, repeats, axis=1) for k, v in start.items()}
      goal = tf.repeat(goal, repeats, axis = 0)

    start = {k: flatten(v) for k, v in start.items()}
    def step(prev, _):
      state, _, _ = prev
      feat = dynamics.get_feat(state)
      inp = tf.stop_gradient(feat) if self._stop_grad_actor else feat
      action = policy(inp, goal).sample()
      succ = dynamics.img_step(state, action, sample=self._config.imag_sample)
      return succ, feat, action
    
    feat = 0 * dynamics.get_feat(start)
    action = policy(feat, goal).mode()
    succ, feats, actions = tools.static_scan(
        step, tf.range(horizon), (start, feat, action))

    states = {k: tf.concat([
        start[k][None], v[:-1]], 0) for k, v in succ.items()}
    if repeats:
      def unfold(tensor):
        s = tensor.shape
        return tf.reshape(tensor, [s[0], s[1] // repeats, repeats] + s[2:])
      states, feats, actions = tf.nest.map_structure(
          unfold, (states, feats, actions))

    return feats, states, actions
