import random
import numpy
import math
import torch
import models
from util import MinMaxStats
from network import NetworkOutput
from game import Action

class Node():

  def __init__(self, prior: float):
    self.visit_count = 0
    self.to_play = -1
    self.prior = prior
    self.value_sum = 0
    self.children = {}
    self.hidden_state = None
    self.reward = 0

  def expanded(self) -> bool:
    return len(self.children) > 0

  def value(self) -> float:
    if self.visit_count == 0:
      return 0
    return self.value_sum / self.visit_count

  # At the start of each search, we add dirichlet noise to the prior of the root
  # to encourage the search to explore new actions.
  def add_exploration_noise(self, config):
    actions = list(self.children.keys())
    noise = numpy.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
      self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac
  
  # We expand a node using the value, reward and policy prediction obtained from
  # the neural network.
  def expand_node(self, to_play, actions, network_output):
    self.to_play = to_play
    self.hidden_state = network_output.hidden_state
    self.reward = network_output.reward
    policy = {a: math.exp(network_output.policy_logits[a]) for a in actions}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
      self.children[action] = Node(p / policy_sum)


def run_mcts(config, root, action_history, network):
  min_max_stats = MinMaxStats(config.known_bounds)

  for _ in range(config.num_simulations):
    history = action_history.clone()
    node = root
    search_path = [node]

    while node.expanded():
      action, node = select_child(config, node, min_max_stats)
      history.add_action(action)
      search_path.append(node)

    # Inside the search tree we use the dynamics function to obtain the next
    # hidden state given an action and the previous hidden state.
    parent = search_path[-2]
    (value, reward, p_logits, hidden_state) = network.recurrent_inference(parent.hidden_state,
                                                 torch.tensor([[history.last_action().index]]))

    policy_logits = {}
    for i in range(config.action_space_size):
      policy_logits[Action(i)] = p_logits[0][i].item()

    network_output = NetworkOutput(
        models.support_to_scalar(value, config.support_size), 
        models.support_to_scalar(reward, config.support_size), 
        policy_logits,
        hidden_state)

    node.expand_node(history.to_play(), history.action_space(), network_output)

    backpropagate(search_path, network_output.value, history.to_play(),
                  config.discount, min_max_stats)

def softmax_sample(visit_counts, actions, temperature: float):
  if temperature == 0:
    action = actions[numpy.argmax(visit_counts)]
  elif temperature == float("inf"):
    action = numpy.random.choice(actions)
  else:
    visit_count_distribution = visit_counts ** (1 / temperature)
    visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
    action = numpy.random.choice(actions, p=visit_count_distribution)

  return 0, action

def select_action(config, num_moves: int, node: Node, network):
  visit_counts = numpy.array(
      [child.visit_count for action, child in node.children.items()],
      dtype="int32")
  actions = [action for action, child in node.children.items()]

  t = config.visit_softmax_temperature_fn(1)
  _, action = softmax_sample(visit_counts, actions, t)
  return action


# Select the child with the highest UCB score.
def select_child(config, node, min_max_stats):
  _, action, child = max(
      (ucb_score(config, node, child, min_max_stats), action,
       child) for action, child in node.children.items())
  return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(config, parent: Node, child: Node,
              min_max_stats) -> float:
  pb_c = math.log((parent.visit_count + config.pb_c_base + 1) /
                  config.pb_c_base) + config.pb_c_init
  pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

  prior_score = pb_c * child.prior
  if child.visit_count > 0:
    value_score = child.reward + config.discount * min_max_stats.normalize(
        child.value())
  else:
    value_score = 0
  return prior_score + value_score



# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path, value, to_play,
                  discount, min_max_stats):
  for node in reversed(search_path):
    node.value_sum += value if node.to_play == to_play else -value
    node.visit_count += 1
    min_max_stats.update(node.value())

    value = node.reward + discount * value


