import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import numpy as np
import tensorflow as tf 
from tensorflow import keras
import gym
import tensorflow_probability as tfp
import tensorflow.keras.losses as kls
from rl_agent import RLAgent
import pybullet_utils.mpi_util as MPIUtil
from pybullet_envs.deep_mimic.env.env import Env
from custom_reward import getRewardCustom
from pybullet_envs.deep_mimic.env.action_space import ActionSpace
from pybullet_envs.deep_mimic.env.pybullet_deep_mimic_env import PyBulletDeepMimicEnv

from pybullet_utils.arg_parser import ArgParser
from pybullet_utils.logger import Logger

import os
import json

save_name = ''

class RLWorld(object):

  def __init__(self, env, arg_parser):
    # TFUtil.disable_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

    self.env = env
    self.arg_parser = arg_parser
    self._enable_training = True
    self.train_agents = []
    self.parse_args(arg_parser)

    self.build_agents()

    return

  def get_enable_training(self):
    return self._enable_training

  def set_enable_training(self, enable):
    self._enable_training = enable
    for i in range(len(self.agents)):
      curr_agent = self.agents[i]
      if curr_agent is not None:
        enable_curr_train = self.train_agents[i] if (len(self.train_agents) > 0) else True
        curr_agent.enable_training = self.enable_training and enable_curr_train

    if (self._enable_training):
      self.env.set_mode(RLAgent.Mode.TRAIN)
    else:
      self.env.set_mode(RLAgent.Mode.TEST)

    return

  enable_training = property(get_enable_training, set_enable_training)

  def parse_args(self, arg_parser):
    self.train_agents = self.arg_parser.parse_bools('train_agents')
    assert (len(self.train_agents) == 1 or len(self.train_agents) == 0)

    return

  def shutdown(self):
    self.env.shutdown()
    return

  def build_agents(self):
    self.agents = []

    Logger.print2('')
    Logger.print2('Num Agents: {:d}'.format(1))

    agent_files = self.arg_parser.parse_strings('agent_files')
    print("len(agent_files)=", len(agent_files))
    assert (len(agent_files) == 1 or len(agent_files) == 0)

    model_files = self.arg_parser.parse_strings('model_files')
    assert (len(model_files) == 1 or len(model_files) == 0)

    
    curr_file = agent_files[0]
    curr_agent = self._build_agent(0, curr_file)

    if curr_agent is not None:
      Logger.print2(str(curr_agent))

      if (len(model_files) > 0):
        curr_model_file = model_files[0]
        if curr_model_file != 'none':
          curr_agent.load_model(os.getcwd() + "/" + curr_model_file)

    self.agents.append(curr_agent)
    Logger.print2('')

    self.set_enable_training(self.enable_training)

    return

  def update(self, timestep):
    self._update_env(timestep)

    # compute next state
    next_state = self.env._humanoid.getState()
    
    # compute reward
    kinPose = self.env._humanoid.computePose(self.env._humanoid._frameFraction)
    reward = getRewardCustom(kinPose,self.env._humanoid)

    # compute whether episode is done
    is_done = self.env.is_episode_end()
    return next_state, reward, is_done

  def reset(self):
    self._reset_agents()
    self._reset_env()
    return

  def end_episode(self):
    self._end_episode_agents()
    return

  def _update_env(self, timestep):
    self.env.update(timestep)
    return

  def _reset_env(self):
    self.env.reset()
    return

  def _reset_agents(self):
    for agent in self.agents:
      if (agent != None):
        agent.reset()
    return

  def _end_episode_agents(self):
    for agent in self.agents:
      if (agent != None):
        agent.end_episode()
    return

  def _build_agent(self, id, agent_file):
    Logger.print2('Agent {:d}: {}'.format(id, agent_file))
    if (agent_file == 'none'):
      agent = None
    else:
      AGENT_TYPE_KEY = "AgentType"
      agent = None
      with open(os.getcwd() + "/" + agent_file) as data_file:
        json_data = json.load(data_file)

        assert AGENT_TYPE_KEY in json_data
        agent_type = json_data[AGENT_TYPE_KEY]

        agent = CustomAgent(self, id, json_data)
        
      assert (agent != None), 'Failed to build agent {:d} from: {}'.format(id, agent_file)
    return agent


class custom_critic(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.d1 = tf.keras.layers.Dense(1024,activation='relu')
    self.d2 = tf.keras.layers.Dense(512,activation='relu')
    # TODO: there should be another layer here 
    self.v = tf.keras.layers.Dense(1, activation = None)

  def call(self, input_data):
    x = self.d1(input_data)
    layer2 = self.d2(x)
    v = self.v(layer2)
    return v
    

class custom_actor(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.num_actions = 36
    self.d1 = tf.keras.layers.Dense(1024,activation='relu')
    self.d2 = tf.keras.layers.Dense(512, activation='relu')
    self.a = tf.keras.layers.Dense(self.num_actions, activation='softmax') # should this have softmax? The paper says it shouldn't

  def call(self, input_data):
    # print("input_data",input_data)
    layer1 = self.d1(input_data)
    # print("layer1",layer1) #### This is becoming all NAN
    layer2 = self.d2(layer1)
    a = self.a(layer2)
    return a

class CustomAgent(RLAgent):
    def __init__(self, world, id, json_data, gamma = 0.95):
        
        super().__init__(world, id, json_data)
        self.world = world
        self.id = id
        self.gamma = gamma
        self.discount_factor = 0.95
        
        self.a_opt = tf.keras.optimizers.SGD(learning_rate=0.00005,momentum=0.9) # policy step size of α(π) = 5 × 10^(−5)
        self.c_opt = tf.keras.optimizers.SGD(learning_rate=0.01,momentum=0.9) #  value stepsize of α(v) = 10^(−2)
        self.actor = custom_actor()
        self.critic = custom_critic()
        self.clip_pram = 0.2

        self.state_size = 197
        self.num_actions = 36

    def load_model(self, in_path):
      self.actor = keras.models.load_model(in_path)

    def act(self,state):
        action = self.actor(np.array([state]))
        # prob = self.actor(np.array([state]))
        # prob = prob.numpy()
        # dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        # action = dist.sample()
        # return int(action.numpy()[0])
        return action.numpy()[0]
    
    # TODO: WE NEED TO ADJUST THIS LOSS FUNCTION TO MATCH THE PAPER
    def actor_loss(self, probs, actions, adv, old_probs, closs):
        
        probability = probs      
        entropy = tf.reduce_mean(tf.math.negative(tf.math.multiply(probability,tf.math.log(probability))))
        #print(probability)
        #print(entropy)
        sur1 = []
        sur2 = []
        
        for pb, t, op in zip(probability, adv, old_probs):
                        t =  tf.constant(t)
                        op =  tf.constant(op)
                        #print(f"t{t}")
                        #ratio = tf.math.exp(tf.math.log(pb + 1e-10) - tf.math.log(op + 1e-10))
                        ratio = tf.math.divide(pb,op)
                        #print(f"ratio{ratio}")
                        s1 = tf.math.multiply(ratio,t)
                        #print(f"s1{s1}")
                        s2 =  tf.math.multiply(tf.clip_by_value(ratio, 1.0 - self.clip_pram, 1.0 + self.clip_pram),t)
                        #print(f"s2{s2}")
                        sur1.append(s1)
                        sur2.append(s2)

        sr1 = tf.stack(sur1)
        sr2 = tf.stack(sur2)
        
        loss = tf.math.negative(tf.reduce_mean(tf.math.minimum(sr1, sr2)) - closs + 0.001 * entropy)
        # loss = tf.reduce_mean(tf.math.minimum(sr1, sr2)) - closs + 0.001 * entropy

        #print(loss)
        return loss

    def learn(self, states, actions,  adv , old_probs, discnt_rewards):
        discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))
        adv = tf.reshape(adv, (len(adv),))

        old_p = old_probs

        old_p = tf.reshape(old_p, (len(old_p),self.num_actions))
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor(states, training=True)
            v =  self.critic(states,training=True)
            v = tf.reshape(v, (len(v),))
            td = tf.math.subtract(discnt_rewards, v)
            c_loss = 0.5 * kls.mean_squared_error(discnt_rewards, v)
            # TODO: We need to figure out this loss function for the actor
            a_loss = self.actor_loss(p, actions, adv, old_probs, c_loss)
            
        grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
        return a_loss, c_loss

def test_reward(env):
  steps_update_world = 0
  total_reward = 0
  world.end_episode()
  world.reset()
  total_reward_count = 0
  # state = env.reset()
  state = env._humanoid.getState()
  done = False
  while not done:
    action = agentoo7.act(state)
    # print("ACTION (IN TEST_REWARD): ", action)
    # take a step with the environment 
    agentoo7._apply_action(action)
    next_state, reward, done = update_world(world)

    state = next_state
    total_reward_count += reward
    done = world.env.is_episode_end()
  return total_reward_count

# This is the GAE function. It measures how much better off the model can be by taking 
# a particular action when in a particular state. It uses the rewards that we collected
# at each time step and calculates how much of an advantage we were able to obtain by 
# taking the action that we took. 
# 
# So if we took a good action, we want to calculate how much better off we were by taking 
# that action, not only in the short run but also over a longer period of time. This way, 
# even if we do not get good rewards in the next time step after the action we tool, 
# we still look at few time steps after that action into the longer future to see how 
# out model performed in the long run
def advantage_estimation(states, actions, rewards, done, values, discount_factor, gamma):
    # initialize advantage and empty list
    gae = 0
    returns = []
    # loop backwards through the rewards
    for i in reversed(range(len(rewards))):
       # Define delta
       # The "done" variable can be thought of as a mask value. If the episode is over then
       # the next state in the batch will be from a newly restarted game so we do not want 
       # to consider that and therefore mask value is taken as 0
       delta = rewards[i] + gamma * values[i + 1] * done[i] - values[i]
       # update gae
       gae = delta + gamma * discount_factor * dones[i] * gae
       # append the return to the list of returns
       returns.append(gae + values[i])

    # reverse the list of returns to restore the original order
    returns.reverse()
    adv = np.array(returns, dtype=np.float32) - values[:-1]
    adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int32)
    returns = np.array(returns, dtype=np.float32)
    return states, actions, returns, adv    

total_reward = 0
steps_update_world = 0
update_timestep = 1. / 240.

def update_world(world):
  next_state, reward, is_done = world.update(update_timestep)
  
  # reward = world.env.calc_reward(agent_id=0)
  global total_reward
  total_reward += reward
  global steps_update_world
  steps_update_world+=1
  
  end_episode = world.env.is_episode_end()
  if (end_episode or steps_update_world>= 1000):
    print("total_reward adfkjdkfsdf =",total_reward)
    total_reward=0
    steps_update_world = 0
    world.end_episode()
    world.reset()
  return next_state, reward, is_done

def build_arg_parser(args):
  arg_parser = ArgParser()
  arg_parser.load_args(args)

  arg_file = arg_parser.parse_string('arg_file', '')
  if arg_file == '':
    arg_file = "run_humanoid3d_backflip_args.txt"
  if (arg_file != ''):
    path = os.getcwd() + "/args/" + arg_file
    succ = arg_parser.load_file(path)
    Logger.print2(arg_file)
    assert succ, Logger.print2('Failed to load args from: ' + arg_file)
  
  # update global variable to use when saving the models
  global save_name
  save_name = arg_file.replace('run_humanoid3d_','').replace('train_humanoid3d_','').replace('_args.txt','')
  return arg_parser

def build_world(args, enable_draw):
  arg_parser = build_arg_parser(args)
  print("enable_draw=", enable_draw)
  env = PyBulletDeepMimicEnv(arg_parser, enable_draw)
  world = RLWorld(env, arg_parser)
  #world.env.set_playback_speed(playback_speed)

  motion_file = arg_parser.parse_string("motion_file")
  print("motion_file build=", motion_file)
  bodies = arg_parser.parse_ints("fall_contact_bodies")
  print("bodies=", bodies)
  agent_files = os.getcwd() + "/" + arg_parser.parse_string("agent_files")

  AGENT_TYPE_KEY = "AgentType"

  print("agent_file=", agent_files)
  with open(agent_files) as data_file:
    json_data = json.load(data_file)
    print("json_data=", json_data)
    assert AGENT_TYPE_KEY in json_data
    agent_type = json_data[AGENT_TYPE_KEY]
    print("agent_type=", agent_type)
    # agent = CustomAgent(world, id, json_data)
    agent = Agent(world, id, json_data)

    agent.set_enable_training(False)
    world.reset()
  return world

if __name__ == '__main__':
  args = sys.argv[1:]
  enable_draw = False
  world = build_world(args, enable_draw)

  env = world.env

  tf.random.set_seed(336699)
  agentoo7 = world.agents[0]
  ppo_steps = 4096
  mini_batches = 256
  ep_reward = []
  total_avgr = []
  target_reached = False 
  best_reward = 0
  avg_rewards_list = []

  while not target_reached:
  # for s in range(ppo_steps):
    if target_reached == True:
            break
    
    done = False
    state = env._humanoid.getState() #env.reset()
    all_aloss = []
    all_closs = []
    rewards = []
    states = []
    actions = []
    probs = []
    dones = []
    values = []
    print("new episod")

    # for e in range(mini_batches):
    for s in range(ppo_steps):
      # world.reset()
      # print("STATE: ", state)
      action = agentoo7.act(state)
      # print("action was: ", action)
      # print("action type: ", type(action))
      value = agentoo7.critic(np.array([state])).numpy()
      # take a step with the environment 
      agentoo7._apply_action(action)
      next_state, reward, done = update_world(world)

      # next_state, reward, done, _ = env.step(action)
      dones.append(1-done)
      rewards.append(reward)
      states.append(state)
      #actions.append(tf.one_hot(action, 2, dtype=tf.int32).numpy().tolist())
      actions.append(action)


      ########################### THIS LINE THAT CALLS PROB IS WRONG ###########
      # The action from the policy specifies target orientations for PD controllers
      # at each joint. IT DOES NOT SPECIFY PROBABILITIES!
      prob = agentoo7.actor(np.array([state]))
      probs.append(prob[0])
      values.append(value[0][0])
      state = next_state
    
    value = agentoo7.critic(np.array([state])).numpy()
    values.append(value[0][0])
    np.reshape(probs, (len(probs),agentoo7.num_actions))
    probs = np.stack(probs, axis=0)

    states, actions,returns, adv  = advantage_estimation(states, actions, rewards, dones, values, agentoo7.discount_factor, agentoo7.gamma)

    ## Update the gradients
    al,cl = agentoo7.learn(states, actions, adv, probs, returns)

    avg_reward = np.mean([test_reward(env) for _ in range(5)])
    print(f"TEST REWARD is {avg_reward}")
    avg_rewards_list.append(avg_reward)
    if avg_reward > best_reward:
          print('Saving Model -- reward improved to: ' + str(avg_reward))
          agentoo7.actor.save('Saved_Models/{}_model_actor'.format(save_name), save_format='tf')
          agentoo7.critic.save('Saved_Models/{}_model_critic'.format(save_name), save_format='tf')
          best_reward = avg_reward
    if best_reward == 200:
          target_reached = True
    # Reset the environment and the humanoid
    total_reward = 0
    steps_update_world = 0
    world.end_episode()
    world.reset()

  env.close()

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda

import argparse
import numpy as np

tf.keras.backend.set_floatx('float64')

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--update_interval', type=int, default=5)
parser.add_argument('--actor_lr', type=float, default=0.0005)
parser.add_argument('--critic_lr', type=float, default=0.001)
parser.add_argument('--clip_ratio', type=float, default=0.2)
parser.add_argument('--lmbda', type=float, default=0.95)
parser.add_argument('--epochs', type=int, default=3)

args = parser.parse_args()


class Actor:
    def __init__(self, state_dim, action_dim, action_bound, std_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.std_bound = std_bound
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(args.actor_lr)

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        mu, std = self.model.predict(state)
        action = np.random.normal(mu[0], std[0], size=self.action_dim)
        action = np.clip(action, -self.action_bound, self.action_bound)
        log_policy = self.log_pdf(mu, std, action)

        return log_policy, action

    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / \
            var - 0.5 * tf.math.log(var * 2 * np.pi)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    def create_model(self):
        state_input = Input((self.state_dim,))
        dense_1 = Dense(32, activation='relu')(state_input)
        dense_2 = Dense(32, activation='relu')(dense_1)
        out_mu = Dense(self.action_dim, activation='tanh')(dense_2)
        mu_output = Lambda(lambda x: x * self.action_bound)(out_mu)
        std_output = Dense(self.action_dim, activation='softplus')(dense_2)
        return tf.keras.models.Model(state_input, [mu_output, std_output])

    def compute_loss(self, log_old_policy, log_new_policy, actions, gaes):
        ratio = tf.exp(log_new_policy - tf.stop_gradient(log_old_policy))
        gaes = tf.stop_gradient(gaes)
        clipped_ratio = tf.clip_by_value(
            ratio, 1.0-args.clip_ratio, 1.0+args.clip_ratio)
        surrogate = -tf.minimum(ratio * gaes, clipped_ratio * gaes)
        return tf.reduce_mean(surrogate)

    def train(self, log_old_policy, states, actions, gaes):
        with tf.GradientTape() as tape:
            mu, std = self.model(states, training=True)
            log_new_policy = self.log_pdf(mu, std, actions)
            loss = self.compute_loss(
                log_old_policy, log_new_policy, actions, gaes)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class Critic:
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(args.critic_lr)

    def create_model(self):
        return tf.keras.Sequential([
            Input((self.state_dim,)),
            Dense(32, activation='relu'),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])

    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def train(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class Agent(RLAgent):
    def __init__(self, env):
        self.env = env
        self.state_dim = 197
        self.action_dim = 36
        self.action_bound = 0.2
        self.std_bound = [1e-2, 1.0]

        self.actor_opt = tf.keras.optimizers.Adam(args.actor_lr)
        self.critic_opt = tf.keras.optimizers.Adam(args.critic_lr)
        self.actor = Actor(self.state_dim, self.action_dim,
                           self.action_bound, self.std_bound)
        self.critic = Critic(self.state_dim)

    def gae_target(self, rewards, v_values, next_v_value, done):
        n_step_targets = np.zeros_like(rewards)
        gae = np.zeros_like(rewards)
        gae_cumulative = 0
        forward_val = 0

        if not done:
            forward_val = next_v_value

        for k in reversed(range(0, len(rewards))):
            delta = rewards[k] + args.gamma * forward_val - v_values[k]
            gae_cumulative = args.gamma * args.lmbda * gae_cumulative + delta
            gae[k] = gae_cumulative
            forward_val = v_values[k]
            n_step_targets[k] = gae[k] + v_values[k]
        return gae, n_step_targets

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

    def train(self, max_episodes=1000):
        for ep in range(max_episodes):
            state_batch = []
            action_batch = []
            reward_batch = []
            old_policy_batch = []

            episode_reward, done = 0, False

            state = self.env.reset()

            while not done:
                # self.env.render()
                log_old_policy, action = self.actor.get_action(state)

                next_state, reward, done, _ = self.env.step(action)

                state = np.reshape(state, [1, self.state_dim])
                action = np.reshape(action, [1, 1])
                next_state = np.reshape(next_state, [1, self.state_dim])
                reward = np.reshape(reward, [1, 1])
                log_old_policy = np.reshape(log_old_policy, [1, 1])

                state_batch.append(state)
                action_batch.append(action)
                reward_batch.append((reward+8)/8)
                old_policy_batch.append(log_old_policy)

                if len(state_batch) >= args.update_interval or done:
                    states = self.list_to_batch(state_batch)
                    actions = self.list_to_batch(action_batch)
                    rewards = self.list_to_batch(reward_batch)
                    old_policys = self.list_to_batch(old_policy_batch)

                    v_values = self.critic.model.predict(states)
                    next_v_value = self.critic.model.predict(next_state)

                    gaes, td_targets = self.gae_target(
                        rewards, v_values, next_v_value, done)

                    for epoch in range(args.epochs):
                        actor_loss = self.actor.train(
                            old_policys, states, actions, gaes)
                        critic_loss = self.critic.train(states, td_targets)

                    state_batch = []
                    action_batch = []
                    reward_batch = []
                    old_policy_batch = []

                episode_reward += reward[0][0]
                state = next_state[0]

            print('EP{} EpisodeReward={}'.format(ep, episode_reward))


def main():
    agent = Agent(env)
    agent.train()
    
    args = sys.argv[1:]
    enable_draw = False
    world = build_world(args, enable_draw)

    env = world.env

    tf.random.set_seed(336699)
    agentoo7 = world.agents[0]
    ppo_steps = 4096
    mini_batches = 256
    ep_reward = []
    total_avgr = []
    target_reached = False 
    best_reward = 0
    avg_rewards_list = []


if __name__ == "__main__":
    main()
  