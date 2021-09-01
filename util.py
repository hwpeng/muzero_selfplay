from network import Network
from typing import Dict, List, Optional
import collections

MAXIMUM_FLOAT_VALUE = float('inf')

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])


class MinMaxStats(object):
  """A class that holds the min-max values of the tree."""

  def __init__(self, known_bounds: Optional[KnownBounds]):
    self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
    self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

  def update(self, value: float):
    self.maximum = max(self.maximum, value)
    self.minimum = min(self.minimum, value)

  def normalize(self, value: float) -> float:
    if self.maximum > self.minimum:
      # We normalize only when we have set the maximum and minimum values.
      return (value - self.minimum) / (self.maximum - self.minimum)
    return value


###############################################

class Player(object):
  pass

###############################################
class SharedStorage():

  def __init__(self):
    self.networks = {}

  def latest_network(self):
    if self.networks:
      return self.networks[max(self.networks.keys())]['weights']
    else:
      # policy -> uniform, value -> 0, reward -> 0
      return make_uniform_network()

def make_uniform_network():
  return Network()

###############################################
class ReplayBuffer():

  def __init__(self, config):
    self.window_size = config.window_size
    #  self.batch_size = config.batch_size
    self.buffer = []

  # Now just for save
  def save_game(self, game):
    if len(self.buffer) > self.window_size:
      self.buffer.pop(0)
    self.buffer.append(game)

