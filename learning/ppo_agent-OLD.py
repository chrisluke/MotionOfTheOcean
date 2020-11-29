import numpy as np
import copy as copy
try:
  import tensorflow.compat.v1 as tf
except Exception:
  import tensorflow as tf

import time
from pybullet_envs.deep_mimic.learning.pg_agent import PGAgent
from learning.mpi_solver import MPISolver
import learning.tf_util as TFUtil
from pybullet_utils.logger import Logger
import pybullet_utils.mpi_util as MPIUtil
import pybullet_utils.math_util as MathUtil
from pybullet_envs.deep_mimic.env.env import Env
from custom_reward import getRewardCustom

'''
Proximal Policy Optimization Agent
'''


class PPOAgent(PGAgent):
  NAME = "PPO"
  EPOCHS_KEY = "Epochs"
  BATCH_SIZE_KEY = "BatchSize"
  RATIO_CLIP_KEY = "RatioClip"
  NORM_ADV_CLIP_KEY = "NormAdvClip"
  TD_LAMBDA_KEY = "TDLambda"
  TAR_CLIP_FRAC = "TarClipFrac"
  ACTOR_STEPSIZE_DECAY = "ActorStepsizeDecay"

  def __init__(self, world, id, json_data):

    super().__init__(world, id, json_data)
    self._exp_action = False
    self.tf_scope = 'agent'
    # self.graph = tf.Graph()
    # self.sess = tf.Session(graph=self.graph)

    return

  def _load_params(self, json_data):
    super()._load_params(json_data)

    self.epochs = 1 if (self.EPOCHS_KEY not in json_data) else json_data[self.EPOCHS_KEY]
    self.batch_size = 1024 if (
        self.BATCH_SIZE_KEY not in json_data) else json_data[self.BATCH_SIZE_KEY]
    self.ratio_clip = 0.2 if (
        self.RATIO_CLIP_KEY not in json_data) else json_data[self.RATIO_CLIP_KEY]
    self.norm_adv_clip = 5 if (
        self.NORM_ADV_CLIP_KEY not in json_data) else json_data[self.NORM_ADV_CLIP_KEY]
    self.td_lambda = 0.95 if (
        self.TD_LAMBDA_KEY not in json_data) else json_data[self.TD_LAMBDA_KEY]
    self.tar_clip_frac = -1 if (
        self.TAR_CLIP_FRAC not in json_data) else json_data[self.TAR_CLIP_FRAC]
    self.actor_stepsize_decay = 0.5 if (
        self.ACTOR_STEPSIZE_DECAY not in json_data) else json_data[self.ACTOR_STEPSIZE_DECAY]

    num_procs = MPIUtil.get_num_procs()
    local_batch_size = int(self.batch_size / num_procs)
    min_replay_size = 2 * local_batch_size  # needed to prevent buffer overflow
    assert (self.replay_buffer_size > min_replay_size)

    self.replay_buffer_size = np.maximum(min_replay_size, self.replay_buffer_size)

    return

  def save_model(self, out_path):
    with self.sess.as_default(), self.graph.as_default():
      try:
        save_path = self.saver.save(self.sess, out_path, write_meta_graph=False, write_state=False)
        Logger.print2('Model saved to: ' + save_path)
      except:
        Logger.print2("Failed to save model to: " + save_path)
    return

  def load_model(self, in_path):
    with self.sess.as_default(), self.graph.as_default():
      self.saver.restore(self.sess, in_path)
      self._load_normalizers()
      Logger.print2('Model loaded from: ' + in_path)
    return

  def _build_nets(self, json_data):
    assert self.ACTOR_NET_KEY in json_data
    assert self.CRITIC_NET_KEY in json_data

    actor_net_name = json_data[self.ACTOR_NET_KEY]
    critic_net_name = json_data[self.CRITIC_NET_KEY]
    actor_init_output_scale = 1 if (self.ACTOR_INIT_OUTPUT_SCALE_KEY not in json_data
                                   ) else json_data[self.ACTOR_INIT_OUTPUT_SCALE_KEY]

    s_size = self.get_state_size()
    g_size = self.get_goal_size()
    a_size = self.get_action_size()

    # setup input tensors
    self.s_tf = tf.placeholder(tf.float32, shape=[None, s_size], name="s")
    self.a_tf = tf.placeholder(tf.float32, shape=[None, a_size], name="a")
    self.tar_val_tf = tf.placeholder(tf.float32, shape=[None], name="tar_val")
    self.adv_tf = tf.placeholder(tf.float32, shape=[None], name="adv")
    self.g_tf = tf.placeholder(tf.float32,
                               shape=([None, g_size] if self.has_goal() else None),
                               name="g")
    self.old_logp_tf = tf.placeholder(tf.float32, shape=[None], name="old_logp")
    self.exp_mask_tf = tf.placeholder(tf.float32, shape=[None], name="exp_mask")

    with tf.variable_scope('main'):
      with tf.variable_scope('actor'):
        self.a_mean_tf = self._build_net_actor(actor_net_name, actor_init_output_scale)
      with tf.variable_scope('critic'):
        self.critic_tf = self._build_net_critic(critic_net_name)

    if (self.a_mean_tf != None):
      Logger.print2('Built actor net: ' + actor_net_name)

    if (self.critic_tf != None):
      Logger.print2('Built critic net: ' + critic_net_name)

    self.norm_a_std_tf = self.exp_params_curr.noise * tf.ones(a_size)
    norm_a_noise_tf = self.norm_a_std_tf * tf.random_normal(shape=tf.shape(self.a_mean_tf))
    norm_a_noise_tf *= tf.expand_dims(self.exp_mask_tf, axis=-1)
    self.sample_a_tf = self.a_mean_tf + norm_a_noise_tf * self.a_norm.std_tf
    self.sample_a_logp_tf = TFUtil.calc_logp_gaussian(x_tf=norm_a_noise_tf,
                                                      mean_tf=None,
                                                      std_tf=self.norm_a_std_tf)

    return
  
  def _build_net_actor(self, net_name, init_output_scale):
        norm_s_tf = self.s_norm.normalize_tf(self.s_tf)
        input_tfs = [norm_s_tf]
        if (self.has_goal()):
            norm_g_tf = self.g_norm.normalize_tf(self.g_tf)
            input_tfs += [norm_g_tf]
        
        h = build_net(net_name, input_tfs)
        norm_a_tf = tf.layers.dense(inputs=h, units=self.get_action_size(), activation=None,
                                kernel_initializer=tf.random_uniform_initializer(minval=-init_output_scale, maxval=init_output_scale))
        
        a_tf = self.a_norm.unnormalize_tf(norm_a_tf)
        return a_tf
    
  def _build_net_critic(self, net_name):
        norm_s_tf = self.s_norm.normalize_tf(self.s_tf)
        input_tfs = [norm_s_tf]
        if (self.has_goal()):
            norm_g_tf = self.g_norm.normalize_tf(self.g_tf)
            input_tfs += [norm_g_tf]
        
        h = build_net(net_name, input_tfs)
        norm_val_tf = tf.layers.dense(inputs=h, units=1, activation=None,
                                kernel_initializer=TFUtil.xavier_initializer);

        norm_val_tf = tf.reshape(norm_val_tf, [-1])
        val_tf = self.val_norm.unnormalize_tf(norm_val_tf)
        return val_tf

  def _build_losses(self, json_data):
    actor_weight_decay = 0 if (
        self.ACTOR_WEIGHT_DECAY_KEY not in json_data) else json_data[self.ACTOR_WEIGHT_DECAY_KEY]
    critic_weight_decay = 0 if (
        self.CRITIC_WEIGHT_DECAY_KEY not in json_data) else json_data[self.CRITIC_WEIGHT_DECAY_KEY]

    norm_val_diff = self.val_norm.normalize_tf(self.tar_val_tf) - self.val_norm.normalize_tf(
        self.critic_tf)
    self.critic_loss_tf = 0.5 * tf.reduce_mean(tf.square(norm_val_diff))

    if (critic_weight_decay != 0):
      self.critic_loss_tf += critic_weight_decay * self._weight_decay_loss('main/critic')

    norm_tar_a_tf = self.a_norm.normalize_tf(self.a_tf)
    self._norm_a_mean_tf = self.a_norm.normalize_tf(self.a_mean_tf)

    self.logp_tf = TFUtil.calc_logp_gaussian(norm_tar_a_tf, self._norm_a_mean_tf,
                                             self.norm_a_std_tf)
    ratio_tf = tf.exp(self.logp_tf - self.old_logp_tf)
    actor_loss0 = self.adv_tf * ratio_tf
    actor_loss1 = self.adv_tf * tf.clip_by_value(ratio_tf, 1.0 - self.ratio_clip,
                                                 1 + self.ratio_clip)
    self.actor_loss_tf = -tf.reduce_mean(tf.minimum(actor_loss0, actor_loss1))

    norm_a_bound_min = self.a_norm.normalize(self.a_bound_min)
    norm_a_bound_max = self.a_norm.normalize(self.a_bound_max)
    a_bound_loss = TFUtil.calc_bound_loss(self._norm_a_mean_tf, norm_a_bound_min, norm_a_bound_max)
    self.actor_loss_tf += a_bound_loss

    if (actor_weight_decay != 0):
      self.actor_loss_tf += actor_weight_decay * self._weight_decay_loss('main/actor')

    # for debugging
    self.clip_frac_tf = tf.reduce_mean(
        tf.to_float(tf.greater(tf.abs(ratio_tf - 1.0), self.ratio_clip)))

    return
  
  def _weight_decay_loss(self, scope):
    vars = self._tf_vars(scope)
    vars_no_bias = [v for v in vars if 'bias' not in v.name]
    loss = tf.add_n([tf.nn.l2_loss(v) for v in vars_no_bias])
    return loss

  def _build_solvers(self, json_data):
    actor_stepsize = 0.001 if (
        self.ACTOR_STEPSIZE_KEY not in json_data) else json_data[self.ACTOR_STEPSIZE_KEY]
    actor_momentum = 0.9 if (
        self.ACTOR_MOMENTUM_KEY not in json_data) else json_data[self.ACTOR_MOMENTUM_KEY]
    critic_stepsize = 0.01 if (
        self.CRITIC_STEPSIZE_KEY not in json_data) else json_data[self.CRITIC_STEPSIZE_KEY]
    critic_momentum = 0.9 if (
        self.CRITIC_MOMENTUM_KEY not in json_data) else json_data[self.CRITIC_MOMENTUM_KEY]

    critic_vars = self._tf_vars('main/critic')
    critic_opt = tf.train.MomentumOptimizer(learning_rate=critic_stepsize,
                                            momentum=critic_momentum)
    self.critic_grad_tf = tf.gradients(self.critic_loss_tf, critic_vars)
    self.critic_solver = MPISolver(self.sess, critic_opt, critic_vars)

    self._actor_stepsize_tf = tf.get_variable(dtype=tf.float32,
                                              name='actor_stepsize',
                                              initializer=actor_stepsize,
                                              trainable=False)
    self._actor_stepsize_ph = tf.get_variable(dtype=tf.float32, name='actor_stepsize_ph', shape=[])
    self._actor_stepsize_update_op = self._actor_stepsize_tf.assign(self._actor_stepsize_ph)

    actor_vars = self._tf_vars('main/actor')
    actor_opt = tf.train.MomentumOptimizer(learning_rate=self._actor_stepsize_tf,
                                           momentum=actor_momentum)
    self.actor_grad_tf = tf.gradients(self.actor_loss_tf, actor_vars)
    self.actor_solver = MPISolver(self.sess, actor_opt, actor_vars)

    return

  def _decide_action(self, s, g):
    with self.sess.as_default(), self.graph.as_default():
      self._exp_action = self._enable_stoch_policy() and MathUtil.flip_coin(
          self.exp_params_curr.rate)
      #print("_decide_action._exp_action=",self._exp_action)
      a, logp = self._eval_actor(s, g, self._exp_action)
    return a[0], logp[0]

  def _eval_actor(self, s, g, enable_exp):
    s = np.reshape(s, [-1, self.get_state_size()])
    g = np.reshape(g, [-1, self.get_goal_size()]) if self.has_goal() else None

    feed = {self.s_tf: s, self.g_tf: g, self.exp_mask_tf: np.array([1 if enable_exp else 0])}

    a, logp = self.sess.run([self.sample_a_tf, self.sample_a_logp_tf], feed_dict=feed)
    return a, logp

  def _train_step(self):
    adv_eps = 1e-5

    start_idx = self.replay_buffer.buffer_tail
    end_idx = self.replay_buffer.buffer_head
    assert (start_idx == 0)
    assert (self.replay_buffer.get_current_size() <= self.replay_buffer.buffer_size
           )  # must avoid overflow
    assert (start_idx < end_idx)

    idx = np.array(list(range(start_idx, end_idx)))
    end_mask = self.replay_buffer.is_path_end(idx)
    end_mask = np.logical_not(end_mask)

    vals = self._compute_batch_vals(start_idx, end_idx)
    new_vals = self._compute_batch_new_vals(start_idx, end_idx, vals)

    valid_idx = idx[end_mask]
    exp_idx = self.replay_buffer.get_idx_filtered(self.EXP_ACTION_FLAG).copy()
    num_valid_idx = valid_idx.shape[0]
    num_exp_idx = exp_idx.shape[0]
    exp_idx = np.column_stack([exp_idx, np.array(list(range(0, num_exp_idx)), dtype=np.int32)])

    local_sample_count = valid_idx.size
    global_sample_count = int(MPIUtil.reduce_sum(local_sample_count))
    mini_batches = int(np.ceil(global_sample_count / self.mini_batch_size))

    adv = new_vals[exp_idx[:, 0]] - vals[exp_idx[:, 0]]
    new_vals = np.clip(new_vals, self.val_min, self.val_max)

    adv_mean = np.mean(adv)
    adv_std = np.std(adv)
    adv = (adv - adv_mean) / (adv_std + adv_eps)
    adv = np.clip(adv, -self.norm_adv_clip, self.norm_adv_clip)

    critic_loss = 0
    actor_loss = 0
    actor_clip_frac = 0

    for e in range(self.epochs):
      np.random.shuffle(valid_idx)
      np.random.shuffle(exp_idx)

      for b in range(mini_batches):
        print("you da best")
        batch_idx_beg = b * self._local_mini_batch_size
        batch_idx_end = batch_idx_beg + self._local_mini_batch_size

        critic_batch = np.array(range(batch_idx_beg, batch_idx_end), dtype=np.int32)
        actor_batch = critic_batch.copy()
        critic_batch = np.mod(critic_batch, num_valid_idx)
        actor_batch = np.mod(actor_batch, num_exp_idx)
        shuffle_actor = (actor_batch[-1] < actor_batch[0]) or (actor_batch[-1] == num_exp_idx - 1)

        critic_batch = valid_idx[critic_batch]
        actor_batch = exp_idx[actor_batch]
        critic_batch_vals = new_vals[critic_batch]
        actor_batch_adv = adv[actor_batch[:, 1]]

        critic_s = self.replay_buffer.get('states', critic_batch)
        critic_g = self.replay_buffer.get('goals', critic_batch) if self.has_goal() else None
        curr_critic_loss = self._update_critic(critic_s, critic_g, critic_batch_vals)

        actor_s = self.replay_buffer.get("states", actor_batch[:, 0])
        actor_g = self.replay_buffer.get("goals", actor_batch[:, 0]) if self.has_goal() else None
        actor_a = self.replay_buffer.get("actions", actor_batch[:, 0])
        actor_logp = self.replay_buffer.get("logps", actor_batch[:, 0])
        curr_actor_loss, curr_actor_clip_frac = self._update_actor(actor_s, actor_g, actor_a,
                                                                   actor_logp, actor_batch_adv)

        critic_loss += curr_critic_loss
        actor_loss += np.abs(curr_actor_loss)
        actor_clip_frac += curr_actor_clip_frac

        if (shuffle_actor):
          np.random.shuffle(exp_idx)

    total_batches = mini_batches * self.epochs
    critic_loss /= total_batches
    actor_loss /= total_batches
    actor_clip_frac /= total_batches

    critic_loss = MPIUtil.reduce_avg(critic_loss)
    actor_loss = MPIUtil.reduce_avg(actor_loss)
    actor_clip_frac = MPIUtil.reduce_avg(actor_clip_frac)

    critic_stepsize = self.critic_solver.get_stepsize()
    actor_stepsize = self.update_actor_stepsize(actor_clip_frac)

    self.logger.log_tabular('Critic_Loss', critic_loss)
    self.logger.log_tabular('Critic_Stepsize', critic_stepsize)
    self.logger.log_tabular('Actor_Loss', actor_loss)
    self.logger.log_tabular('Actor_Stepsize', actor_stepsize)
    self.logger.log_tabular('Clip_Frac', actor_clip_frac)
    self.logger.log_tabular('Adv_Mean', adv_mean)
    self.logger.log_tabular('Adv_Std', adv_std)

    self.replay_buffer.clear()

    return

  def update(self, timestep):
    if self.need_new_action():
      # print("update_new_action!!!")
      self._update_new_action()

    if (self._mode == self.Mode.TRAIN and self.enable_training):
      self._update_counter += timestep

      while self._update_counter >= self.update_period:
        with self.sess.as_default(), self.graph.as_default():
          self._train()
        self._update_exp_params()
        self.world.env.set_sample_count(self._total_sample_count)
        self._update_counter -= self.update_period

    return
  
  def _update_new_action(self):
        s = self._record_state()
        g = self._record_goal()

        if not (self._is_first_step()):
            r = self._record_reward()
            self.path.rewards.append(r)
        
        a, logp = self._decide_action(s=s, g=g)
        assert len(np.shape(a)) == 1
        assert len(np.shape(logp)) <= 1

        flags = self._record_flags()
        self._apply_action(a)

        self.path.states.append(s)
        self.path.goals.append(g)
        self.path.actions.append(a)
        self.path.logps.append(logp)
        self.path.flags.append(flags)
        
        if self._enable_draw():
            self._log_val(s, g)
        
        return

  def _train(self):
    samples = self.replay_buffer.total_count
    self._total_sample_count = int(MPIUtil.reduce_sum(samples))
    end_training = False

    if (self.replay_buffer_initialized):
      if (self._valid_train_step()):
        prev_iter = self.iter
        iters = self._get_iters_per_update()
        avg_train_return = MPIUtil.reduce_avg(self.train_return)

        for i in range(iters):
          curr_iter = self.iter
          wall_time = time.time() - self.start_time
          wall_time /= 60 * 60  # store time in hours

          has_goal = self.has_goal()
          s_mean = np.mean(self.s_norm.mean)
          s_std = np.mean(self.s_norm.std)
          g_mean = np.mean(self.g_norm.mean) if has_goal else 0
          g_std = np.mean(self.g_norm.std) if has_goal else 0

          self.logger.log_tabular("Iteration", self.iter)
          self.logger.log_tabular("Wall_Time", wall_time)
          self.logger.log_tabular("Samples", self._total_sample_count)
          self.logger.log_tabular("Train_Return", avg_train_return)
          self.logger.log_tabular("Test_Return", self.avg_test_return)
          self.logger.log_tabular("State_Mean", s_mean)
          self.logger.log_tabular("State_Std", s_std)
          self.logger.log_tabular("Goal_Mean", g_mean)
          self.logger.log_tabular("Goal_Std", g_std)
          self._log_exp_params()

          self._update_iter(self.iter + 1)
          self._train_step()

          Logger.print2("Agent " + str(self.id))
          self.logger.print_tabular()
          Logger.print2("")

          if (self._enable_output() and curr_iter % self.int_output_iters == 0):
            self.logger.dump_tabular()

        if (prev_iter // self.int_output_iters != self.iter // self.int_output_iters):
          end_training = self.enable_testing()

    else:

      Logger.print2("Agent " + str(self.id))
      Logger.print2("Samples: " + str(self._total_sample_count))
      Logger.print2("")

      if (self._total_sample_count >= self.init_samples):
        self.replay_buffer_initialized = True
        end_training = self.enable_testing()

    if self._need_normalizer_update:
      self._update_normalizers()
      self._need_normalizer_update = self.normalizer_samples > self._total_sample_count

    if end_training:
      self._init_mode_train_end()

    return

  def _get_iters_per_update(self):
    return 1

  def _valid_train_step(self):
    samples = self.replay_buffer.get_current_size()
    exp_samples = self.replay_buffer.count_filtered(self.EXP_ACTION_FLAG)
    global_sample_count = int(MPIUtil.reduce_sum(samples))
    global_exp_min = int(MPIUtil.reduce_min(exp_samples))
    return (global_sample_count > self.batch_size) and (global_exp_min > 0)

  def _compute_batch_vals(self, start_idx, end_idx):
    states = self.replay_buffer.get_all("states")[start_idx:end_idx]
    goals = self.replay_buffer.get_all("goals")[start_idx:end_idx] if self.has_goal() else None

    idx = np.array(list(range(start_idx, end_idx)))
    is_end = self.replay_buffer.is_path_end(idx)
    is_fail = self.replay_buffer.check_terminal_flag(idx, Env.Terminate.Fail)
    is_succ = self.replay_buffer.check_terminal_flag(idx, Env.Terminate.Succ)
    is_fail = np.logical_and(is_end, is_fail)
    is_succ = np.logical_and(is_end, is_succ)

    vals = self._eval_critic(states, goals)
    vals[is_fail] = self.val_fail
    vals[is_succ] = self.val_succ

    return vals

  def _compute_batch_new_vals(self, start_idx, end_idx, val_buffer):
    rewards = self.replay_buffer.get_all("rewards")[start_idx:end_idx]

    if self.discount == 0:
      new_vals = rewards.copy()
    else:
      new_vals = np.zeros_like(val_buffer)

      curr_idx = start_idx
      while curr_idx < end_idx:
        idx0 = curr_idx - start_idx
        idx1 = self.replay_buffer.get_path_end(curr_idx) - start_idx
        r = rewards[idx0:idx1]
        v = val_buffer[idx0:(idx1 + 1)]

        new_vals[idx0:idx1] = compute_return(r, self.discount, self.td_lambda, v)
        curr_idx = idx1 + start_idx + 1

    return new_vals

  def _update_critic(self, s, g, tar_vals):
    feed = {self.s_tf: s, self.g_tf: g, self.tar_val_tf: tar_vals}

    loss, grads = self.sess.run([self.critic_loss_tf, self.critic_grad_tf], feed)
    self.critic_solver.update(grads)
    return loss

  def _update_actor(self, s, g, a, logp, adv):
    feed = {self.s_tf: s, self.g_tf: g, self.a_tf: a, self.adv_tf: adv, self.old_logp_tf: logp}

    loss, grads, clip_frac = self.sess.run(
        [self.actor_loss_tf, self.actor_grad_tf, self.clip_frac_tf], feed)
    self.actor_solver.update(grads)

    return loss, clip_frac

  def update_actor_stepsize(self, clip_frac):
    clip_tol = 1.5
    step_scale = 2
    max_stepsize = 1e-2
    min_stepsize = 1e-8
    warmup_iters = 5

    actor_stepsize = self.actor_solver.get_stepsize()
    if (self.tar_clip_frac >= 0 and self.iter > warmup_iters):
      min_clip = self.tar_clip_frac / clip_tol
      max_clip = self.tar_clip_frac * clip_tol
      under_tol = clip_frac < min_clip
      over_tol = clip_frac > max_clip

      if (over_tol or under_tol):
        if (over_tol):
          actor_stepsize *= self.actor_stepsize_decay
        else:
          actor_stepsize /= self.actor_stepsize_decay

        actor_stepsize = np.clip(actor_stepsize, min_stepsize, max_stepsize)
        self.set_actor_stepsize(actor_stepsize)

    return actor_stepsize

  def set_actor_stepsize(self, stepsize):
    feed = {
        self._actor_stepsize_ph: stepsize,
    }
    self.sess.run(self._actor_stepsize_update_op, feed)
    return
  
  def _record_reward(self):
    kinPose = self.world.env._humanoid.computePose(self.world.env._humanoid._frameFraction)
    reward = getRewardCustom(kinPose,self.world.env._humanoid)
    return reward

def compute_return(rewards, gamma, td_lambda, val_t):
  # computes td-lambda return of path
  path_len = len(rewards)
  assert len(val_t) == path_len + 1

  return_t = np.zeros(path_len)
  last_val = rewards[-1] + gamma * val_t[-1]
  return_t[-1] = last_val

  for i in reversed(range(0, path_len - 1)):
    curr_r = rewards[i]
    next_ret = return_t[i + 1]
    curr_val = curr_r + gamma * ((1.0 - td_lambda) * val_t[i + 1] + td_lambda * next_ret)
    return_t[i] = curr_val

  return return_t

def build_net(net_name, input_tfs, reuse=False):
  net = None

  if (net_name == "fc_2layers_1024units"):
    layers = [1024, 512]
    activation = tf.nn.relu

    input_tf = tf.concat(axis=-1, values=input_tfs)
    net = TFUtil.fc_net(input_tf, layers, activation=activation, reuse=reuse)
    net = activation(net)
  else:
    assert False, 'Unsupported net: ' + net_name

  return net