import numpy as np

# TODO: delete tensorflow.compat.v1 import --> change to new tensorflow version
try:
  import tensorflow.compat.v1 as tf
except Exception:
  import tensorflow as tf

from pybullet_envs.deep_mimic.learning.tf_normalizer import TFNormalizer
import pybullet_envs.deep_mimic.learning.rl_util as RLUtil
from pybullet_envs.deep_mimic.env.action_space import ActionSpace
import copy as copy

import time
from learning.mpi_solver import MPISolver
import learning.tf_util as TFUtil
from pybullet_utils.logger import Logger
import pybullet_utils.mpi_util as MPIUtil
import pybullet_utils.math_util as MathUtil
from pybullet_envs.deep_mimic.env.env import Env
from custom_reward import getRewardCustom
from pybullet_envs.deep_mimic.learning.tf_agent import RLAgent
'''
Proximal Policy Optimization Agent
'''


class PPOAgent(RLAgent):
  RESOURCE_SCOPE = 'resource'
  SOLVER_SCOPE = 'solvers'
  ACTOR_NET_KEY = 'ActorNet'
  ACTOR_STEPSIZE_KEY = 'ActorStepsize'
  ACTOR_MOMENTUM_KEY = 'ActorMomentum'
  ACTOR_WEIGHT_DECAY_KEY = 'ActorWeightDecay'
  ACTOR_INIT_OUTPUT_SCALE_KEY = 'ActorInitOutputScale'

  CRITIC_NET_KEY = 'CriticNet'
  CRITIC_STEPSIZE_KEY = 'CriticStepsize'
  CRITIC_MOMENTUM_KEY = 'CriticMomentum'
  CRITIC_WEIGHT_DECAY_KEY = 'CriticWeightDecay'

  EXP_ACTION_FLAG = 1 << 0

  NAME = "PPO"
  EPOCHS_KEY = "Epochs"
  BATCH_SIZE_KEY = "BatchSize"
  RATIO_CLIP_KEY = "RatioClip"
  NORM_ADV_CLIP_KEY = "NormAdvClip"
  TD_LAMBDA_KEY = "TDLambda"
  TAR_CLIP_FRAC = "TarClipFrac"
  ACTOR_STEPSIZE_DECAY = "ActorStepsizeDecay"

  def __init__(self, world, id, json_data):
    self.state_size = 197
    self.num_actions = 36
    self.tf_scope = 'agent'
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph)

    super().__init__(world, id, json_data)
    self._build_graph(json_data)
    self._init_normalizers()

    self._exp_action = False
    self.tf_scope = 'agent'
    self.custom_model = ReinforceWithBaseline(self.num_actions)
    

    return
  def __del__(self):
    self.sess.close()
    return

  def _get_output_path(self):
    assert (self.output_dir != '')
    file_path = self.output_dir + '/agent' + str(self.id) + '_model.ckpt'
    return file_path

  def _get_int_output_path(self):
    assert (self.int_output_dir != '')
    file_path = self.int_output_dir + (
        '/agent{:d}_models/agent{:d}_int_model_{:010d}.ckpt').format(self.id, self.id, self.iter)
    return file_path

  def _build_graph(self, json_data):
    with self.sess.as_default(), self.graph.as_default():
      with tf.variable_scope(self.tf_scope):
        self._build_nets(json_data)

        with tf.variable_scope(self.SOLVER_SCOPE):
          self._build_losses(json_data)
          self._build_solvers(json_data)

        self._initialize_vars()
        self._build_saver()
    return


  def _tf_vars(self, scope=''):
    with self.sess.as_default(), self.graph.as_default():
      res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.tf_scope + '/' + scope)
      assert len(res) > 0
    return res

  def _update_normalizers(self):
    with self.sess.as_default(), self.graph.as_default():
      super()._update_normalizers()
    return


  def _build_saver(self):
    vars = self._get_saver_vars()
    self.saver = tf.train.Saver(vars, max_to_keep=0)
    return

  def _get_saver_vars(self):
    with self.sess.as_default(), self.graph.as_default():
      vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.tf_scope)
      vars = [v for v in vars if '/' + self.SOLVER_SCOPE + '/' not in v.name]
      #vars = [v for v in vars if '/target/' not in v.name]
      assert len(vars) > 0
    return vars
  def _check_action_space(self):
    action_space = self.get_action_space()
    return action_space == ActionSpace.Continuous

  def reset(self):
    super().reset()
    self._exp_action = False
    return
  def _build_normalizers(self):
    with self.sess.as_default(), self.graph.as_default(), tf.variable_scope(self.tf_scope):
      with tf.variable_scope(self.RESOURCE_SCOPE):
        self.s_norm = TFNormalizer(self.sess, 's_norm', self.state_size,
                                   self.world.env.build_state_norm_groups(self.id))
        state_offset = -self.world.env.build_state_offset(self.id)
        print("state_offset=", state_offset)
        state_scale = 1 / self.world.env.build_state_scale(self.id)
        print("state_scale=", state_scale)
        self.s_norm.set_mean_std(-self.world.env.build_state_offset(self.id),
                                 1 / self.world.env.build_state_scale(self.id))

        
        

        self.a_norm = TFNormalizer(self.sess, 'a_norm', self.get_action_size())
        self.a_norm.set_mean_std(-self.world.env.build_action_offset(self.id),
                                 1 / self.world.env.build_action_scale(self.id))
    with self.sess.as_default(), self.graph.as_default(), tf.variable_scope(self.tf_scope):
      with tf.variable_scope(self.RESOURCE_SCOPE):
        val_offset, val_scale = self._calc_val_offset_scale(self.discount)
        self.val_norm = TFNormalizer(self.sess, 'val_norm', 1)
        self.val_norm.set_mean_std(-val_offset, 1.0 / val_scale)
    return

  def _init_normalizers(self):
    with self.sess.as_default(), self.graph.as_default():
      # update normalizers to sync the tensorflow tensors
      self.s_norm.update()
      self.a_norm.update()
    with self.sess.as_default(), self.graph.as_default():
      self.val_norm.update()
    return

  def _load_normalizers(self):
    self.s_norm.load()
    self.a_norm.load()
    self.val_norm.load()
    return

  def _initialize_vars(self):
    self.sess.run(tf.global_variables_initializer())
    self._sync_solvers()
    return

  def _sync_solvers(self):
    self.actor_solver.sync()
    self.critic_solver.sync()
    return

  def _enable_stoch_policy(self):
    return self.enable_training and (self._mode == self.Mode.TRAIN or
                                     self._mode == self.Mode.TRAIN_END)

  def _load_params(self, json_data):
    super()._load_params(json_data)
    
    self.val_min, self.val_max = self._calc_val_bounds(self.discount)
    self.val_fail, self.val_succ = self._calc_term_vals(self.discount)

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
  def _eval_critic(self, s):
    with self.sess.as_default(), self.graph.as_default():
      s = np.reshape(s, [-1, self.state_size])
      

      feed = {self.s_tf: s}

      val = self.critic_tf.eval(feed)
    return val
  def _record_flags(self):
    flags = int(0)
    if (self._exp_action):
      flags = flags | self.EXP_ACTION_FLAG
    return flags

  def _build_replay_buffer(self, buffer_size):
    super()._build_replay_buffer(buffer_size)
    self.replay_buffer.add_filter_key(self.EXP_ACTION_FLAG)
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

  # def _build_losses(self, json_data):
  #   actor_weight_decay = 0 if (
  #       self.ACTOR_WEIGHT_DECAY_KEY not in json_data) else json_data[self.ACTOR_WEIGHT_DECAY_KEY]
  #   critic_weight_decay = 0 if (
  #       self.CRITIC_WEIGHT_DECAY_KEY not in json_data) else json_data[self.CRITIC_WEIGHT_DECAY_KEY]

  #   norm_val_diff = self.val_norm.normalize_tf(self.tar_val_tf) - self.val_norm.normalize_tf(
  #       self.critic_tf)
  #   self.critic_loss_tf = 0.5 * tf.reduce_mean(tf.square(norm_val_diff))

  #   if (critic_weight_decay != 0):
  #     self.critic_loss_tf += critic_weight_decay * self._weight_decay_loss('main/critic')

  #   norm_tar_a_tf = self.a_norm.normalize_tf(self.a_tf)
  #   self._norm_a_mean_tf = self.a_norm.normalize_tf(self.a_mean_tf)

  #   self.logp_tf = TFUtil.calc_logp_gaussian(norm_tar_a_tf, self._norm_a_mean_tf,
  #                                            self.norm_a_std_tf)
  #   ratio_tf = tf.exp(self.logp_tf - self.old_logp_tf)
  #   actor_loss0 = self.adv_tf * ratio_tf
  #   actor_loss1 = self.adv_tf * tf.clip_by_value(ratio_tf, 1.0 - self.ratio_clip,
  #                                                1 + self.ratio_clip)
  #   self.actor_loss_tf = -tf.reduce_mean(tf.minimum(actor_loss0, actor_loss1))

  #   norm_a_bound_min = self.a_norm.normalize(self.a_bound_min)
  #   norm_a_bound_max = self.a_norm.normalize(self.a_bound_max)
  #   a_bound_loss = TFUtil.calc_bound_loss(self._norm_a_mean_tf, norm_a_bound_min, norm_a_bound_max)
  #   self.actor_loss_tf += a_bound_loss

  #   if (actor_weight_decay != 0):
  #     self.actor_loss_tf += actor_weight_decay * self._weight_decay_loss('main/actor')

  #   # for debugging
  #   self.clip_frac_tf = tf.reduce_mean(
  #       tf.to_float(tf.greater(tf.abs(ratio_tf - 1.0), self.ratio_clip)))

  #   return
  
  # def _weight_decay_loss(self, scope):
  #   vars = self._tf_vars(scope)
  #   vars_no_bias = [v for v in vars if 'bias' not in v.name]
  #   loss = tf.add_n([tf.nn.l2_loss(v) for v in vars_no_bias])
  #   return loss

  
  def custom_decide_action(self, s):
    logits = self.custom_model.call(np.reshape(s, [-1, self.state_size]))

    action_choices = np.arange(self.num_actions)
    normed_probs = np.linalg.norm(logits, axis=0)
    action_index = np.random.choice(action_choices, 1, p=normed_probs)
    custom_action = action_index[0]

    new_standard_deviation = tf.math.reduce_std(logits)
      
    custom_logp = calc_logp_gaussian(logits,mean_tf=None,std_tf=new_standard_deviation)
    return custom_action, custom_logp

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

    vals = self.custom_model.custom_compute_batch_vals(self,start_idx, end_idx)
    new_vals = self.custom_model.custom_compute_batch_new_vals(start_idx, end_idx, vals)

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
        curr_critic_loss = self._update_critic(critic_s, critic_batch_vals)

        actor_s = self.replay_buffer.get("states", actor_batch[:, 0])
        actor_a = self.replay_buffer.get("actions", actor_batch[:, 0])
        actor_logp = self.replay_buffer.get("logps", actor_batch[:, 0])
        curr_actor_loss, curr_actor_clip_frac = self._update_actor(actor_s, actor_a,
                                                                   actor_logp, actor_batch_adv)

        actor_clip_frac += curr_actor_clip_frac

        if (shuffle_actor):
          np.random.shuffle(exp_idx)

    total_batches = mini_batches * self.epochs
    actor_clip_frac /= total_batches

    actor_clip_frac = MPIUtil.reduce_avg(actor_clip_frac)

    self.logger.log_tabular('Clip_Frac', actor_clip_frac)
    self.logger.log_tabular('Adv_Mean', adv_mean)
    self.logger.log_tabular('Adv_Std', adv_std)

    self.replay_buffer.clear()

    return

  def update(self, timestep):
    if self.need_new_action():
      # print("update_new_action!!!")
      state = self.world.env.record_state(self.id)
      a, logp = self._decide_action(s=state)
      print("a.shape: ", a.shape,  "logp: " , logp)

      custom_a, custom_logp = self.custom_decide_action(s=state)
      
      
      self._update_new_action(state, custom_a, custom_logp)

    if (self._mode == self.Mode.TRAIN and self.enable_training):
      self._update_counter += timestep

      while self._update_counter >= self.update_period:
        with tf.GradientTape() as tape:
          self._train()
        self._update_exp_params()
        self.world.env.set_sample_count(self._total_sample_count)
        self._update_counter -= self.update_period
      

    return
  
  def _update_new_action(self,state,a,logp):
        g = self._record_goal()

        if not (self._is_first_step()):
            r = self._record_reward()
            self.path.rewards.append(r)
        
        
        assert len(np.shape(a)) == 1
        assert len(np.shape(logp)) <= 1

        flags = self._record_flags()
        self._apply_action(a)

        self.path.states.append(state)
        self.path.actions.append(a)
        self.path.goals.append(g)
        self.path.logps.append(logp)
        self.path.flags.append(flags)
        
        return

  def _train(self):
    samples = self.replay_buffer.total_count
    self._total_sample_count = int(MPIUtil.reduce_sum(samples))
    end_training = False

    if (self.replay_buffer_initialized):
      if (self._valid_train_step()):
        prev_iter = self.iter
        iters = 1
        avg_train_return = MPIUtil.reduce_avg(self.train_return)

        for i in range(iters):
          curr_iter = self.iter
          wall_time = time.time() - self.start_time
          wall_time /= 60 * 60  # store time in hours

          has_goal = False
          s_mean = np.mean(self.s_norm.mean)
          s_std = np.mean(self.s_norm.std)

          self.logger.log_tabular("Iteration", self.iter)
          self.logger.log_tabular("Wall_Time", wall_time)
          self.logger.log_tabular("Samples", self._total_sample_count)
          self.logger.log_tabular("Train_Return", avg_train_return)
          self.logger.log_tabular("Test_Return", self.avg_test_return)
          self.logger.log_tabular("State_Mean", s_mean)
          self.logger.log_tabular("State_Std", s_std)
          self._log_exp_params()

          self._update_iter(self.iter + 1)
          self._train_step()

          Logger.print2("Agent " + str(self.id))
          self.logger.print_tabular()
          Logger.print2("")

          if (self._enable_output() and curr_iter % self.int_output_iters == 0):
            # this line writes a log recording what was printed for the curretn iteration
            # to a file called agent0_log.txt in /output folder
            self.logger.dump_tabular()

        if (prev_iter // self.int_output_iters != self.iter // self.int_output_iters):
          end_training = self.enable_testing()

    else:
      print("WENT INTO ELSE")
      Logger.print2("Agent " + str(self.id))
      Logger.print2("Samples: " + str(self._total_sample_count))
      Logger.print2("")

      if (self._total_sample_count >= self.init_samples):
        self.replay_buffer_initialized = True
        end_training = self.enable_testing()

    if self._need_normalizer_update:
      print("UPDATE NORMALIZERS")
      self._update_normalizers()
      self._need_normalizer_update = self.normalizer_samples > self._total_sample_count

    if end_training:
      print("END TRAINING")
      self._init_mode_train_end()

    return

  def _valid_train_step(self):
    samples = self.replay_buffer.get_current_size()
    exp_samples = self.replay_buffer.count_filtered(self.EXP_ACTION_FLAG)
    global_sample_count = int(MPIUtil.reduce_sum(samples))
    global_exp_min = int(MPIUtil.reduce_min(exp_samples))
    return (global_sample_count > self.batch_size) and (global_exp_min > 0)

  def _compute_batch_vals(self, start_idx, end_idx):
    states = self.replay_buffer.get_all("states")[start_idx:end_idx]

    idx = np.array(list(range(start_idx, end_idx)))
    is_end = self.replay_buffer.is_path_end(idx)
    is_fail = self.replay_buffer.check_terminal_flag(idx, Env.Terminate.Fail)
    is_succ = self.replay_buffer.check_terminal_flag(idx, Env.Terminate.Succ)
    is_fail = np.logical_and(is_end, is_fail)
    is_succ = np.logical_and(is_end, is_succ)

    vals = self._eval_critic(states)
    vals[is_fail] = self.val_fail
    vals[is_succ] = self.val_succ

    return vals
  

  def _update_critic(self, s, tar_vals):
    feed = {self.s_tf: s, self.tar_val_tf: tar_vals}

    loss, grads = self.sess.run([self.critic_loss_tf, self.critic_grad_tf], feed)
    self.critic_solver.update(grads)
    return loss

  def _update_actor(self, s, a, logp, adv):
    feed = {self.s_tf: s, self.a_tf: a, self.adv_tf: adv, self.old_logp_tf: logp}

    loss, grads, clip_frac = self.sess.run(
        [self.actor_loss_tf, self.actor_grad_tf, self.clip_frac_tf], feed)
    self.actor_solver.update(grads)

    return loss, clip_frac
  
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


def calc_logp_gaussian(x_tf, mean_tf, std_tf):
  dim = tf.to_float(tf.shape(x_tf)[-1])

  if mean_tf is None:
    diff_tf = x_tf
  else:
    diff_tf = x_tf - mean_tf

  logp_tf = -0.5 * tf.reduce_sum(tf.square(diff_tf / std_tf), axis=-1)
  logp_tf += tf.cast(-0.5 * dim * np.log(2 * np.pi) - tf.cast(std_tf, dtype=tf.float32),dtype=tf.float64)

  return logp_tf

class ReinforceWithBaseline(tf.keras.Model):
    def __init__(self, num_actions):
        """
        The ReinforceWithBaseline class that inherits from tf.keras.Model.

        The forward pass calculates the policy for the agent given a batch of states. During training,
        ReinforceWithBaseLine estimates the value of each state to be used as a baseline to compare the policy's
        performance with.

        :param state_size: number of parameters that define the state. You don't necessarily have to use this, 
                           but it can be used as the input size for your first dense layer.
        :param num_actions: number of actions in an environment
        """
        super(ReinforceWithBaseline, self).__init__()
        self.num_actions = num_actions

        # initial_learning_rate=0.1
        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,decay_steps=1000, decay_rate=0.96)
        
        # learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(1e-4, decay_steps=100, decay_rate=0.96, staircase=False)
        learning_rate = tf.keras.optimizers.schedules.InverseTimeDecay(0.001,10000,0.7)
        # TODO: Define actor network parameters, critic network parameters, and optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.policyLayer1 = tf.keras.layers.Dense(1024, activation='relu')
        self.policyLayer2 = tf.keras.layers.Dense(512, activation='relu')
        self.policyLayer3 = tf.keras.layers.Dense(num_actions, activation='softmax')

        # Value Feed Forward Network
        self.value_net_1 = tf.keras.layers.Dense(1024, activation='relu') 
        self.value_net_2 = tf.keras.layers.Dense(1) # don't apply softmax since the output of the critic network is values

        self.value_stepsize = 0.01 
        self.policy_stepsize = 0.00005
        self.momentum = 0.9 
        self.batch_size = 4096
        self.mini_batch_size = 256
        self.epochs = 1 
        self.norm_adv_clip = 0.2
        self.ratio_clip = 0.2
        self.td_lambda = 0.95
        

    def call(self, states):
        """
        Performs the forward pass on a batch of states to generate the action probabilities.
        This returns a policy tensor of shape [episode_length, num_actions], where each row is a
        probability distribution over actions for each state.

        :param states: An [episode_length, state_size] dimensioned array
        representing the history of states of an episode
        :return: A [episode_length,num_actions] matrix representing the probability distribution over actions
        for each state in the episode
        """
        # TODO: implement this!
        layer1 = self.policyLayer1(states)
        layer2 = self.policyLayer2(layer1)
        logits = self.policyLayer3(layer2)
        return logits

    def value_function(self, states):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An [episode_length, state_size] dimensioned array representing the history of states
        of an episode.
        :return: A [episode_length] matrix representing the value of each state.
        """
        # TODO: implement this :D
        layer1 = self.value_net_1(states)
        layer2 = self.value_net_2(layer1)
        # print("value_function returns: ", layer2)
        return layer2
    
    def custom_compute_batch_vals(self, agent, start_idx, end_idx):
        states = agent.replay_buffer.get_all("states")[start_idx:end_idx]

        idx = np.array(list(range(start_idx, end_idx)))
        is_end = agent.replay_buffer.is_path_end(idx)
        is_fail = agent.replay_buffer.check_terminal_flag(idx, Env.Terminate.Fail)
        is_succ = agent.replay_buffer.check_terminal_flag(idx, Env.Terminate.Succ)
        is_fail = np.logical_and(is_end, is_fail)
        is_succ = np.logical_and(is_end, is_succ)

        layer1 = self.value_net_1(states)
        vals = self.value_net_2(layer1)
        vals[is_fail] = self.val_fail
        vals[is_succ] = self.val_succ

        return vals
    
    def custom_compute_batch_new_vals(self, agent, start_idx, end_idx, val_buffer):
        rewards = agent.replay_buffer.get_all("rewards")[start_idx:end_idx]
        print("val_buffer: ", val_buffer)
        if agent.discount == 0:
          new_vals = rewards.copy()
        else:
          new_vals = np.zeros_like(val_buffer)

          curr_idx = start_idx
          while curr_idx < end_idx:
            idx0 = curr_idx - start_idx
            idx1 = agent.replay_buffer.get_path_end(curr_idx) - start_idx
            r = rewards[idx0:idx1]
            v = val_buffer[idx0:(idx1 + 1)]

            new_vals[idx0:idx1] = compute_return(r, agent.discount, agent.td_lambda, v)
            curr_idx = idx1 + start_idx + 1

        return new_vals

    def loss(self, states, actions, discounted_rewards):
        """
        Computes the loss for the agent. Refer to the lecture slides referenced in the handout to see how this is done.

        Remember that the loss is similar to the loss as in part 1, with a few specific changes.

        1) In your actor loss, instead of element-wise multiplying with discounted_rewards, you want to element-wise multiply with your advantage. 
        See handout/slides for definition of advantage.
        
        2) In your actor loss, you must use tf.stop_gradient on the advantage to stop the loss calculated on the actor network 
        from propagating back to the critic network.
        
        3) See handout/slides for how to calculate the loss for your critic network.

        :param states: A batch of states of shape (episode_length, state_size)
        :param actions: History of actions taken at each timestep of the episode (represented as an [episode_length] array)
        :param discounted_rewards: Discounted rewards throughout a complete episode (represented as an [episode_length] array)
        :return: loss, a TensorFlow scalar
        """
        # TODO: implement this :)
        # Hint: use tf.gather_nd (https://www.tensorflow.org/api_docs/python/tf/gather_nd) to get the probabilities of the actions taken by the model
        
        states = tf.convert_to_tensor(states)
        value_matrix = self.value_function(tf.cast(states,tf.float32))

        discounted_rewards = tf.convert_to_tensor(discounted_rewards)
        advantage = tf.math.subtract(discounted_rewards,tf.squeeze(value_matrix))

        print("advantage: ", advantage)

        ### ACTOR LOSS ##########
        indices = np.arange(len(states))
        actions = tf.convert_to_tensor(np.stack((indices,np.asarray(actions)),axis=1))

        # use the model's call function to get the action probabilities
        policy_matrix = self.call(tf.cast(states,tf.float32))   
        
        action_probabilities = tf.gather_nd(policy_matrix, actions)
        log_probs = tf.math.log(action_probabilities)
        # Call stop_gradient so we don’t backprop through the value network while computing gradients for the actor
        dicount_probs = tf.math.multiply(log_probs, tf.stop_gradient(advantage))
        # sum_tensor = tf.math.reduce_sum(dicount_probs)

        actor_loss = tf.math.multiply(dicount_probs, -1)

        
        ##### CRITIC LOSS #######
        critic_loss = tf.math.square(advantage) 

        final_loss = tf.math.reduce_sum(tf.math.add(actor_loss,critic_loss))
        # print("final_loss: ", final_loss)
        return final_loss

