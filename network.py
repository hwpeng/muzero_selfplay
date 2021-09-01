import typing
from typing import Dict, List, Optional
from game import Action

class NetworkOutput(typing.NamedTuple):
  value: float
  reward: float
  policy_logits: Dict[Action, float]
  hidden_state: List[float]


class Network(object):

  def initial_inference(self, image) -> NetworkOutput:
    # representation + prediction function
    return NetworkOutput(0, 0, {}, [])

  def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
    # dynamics + prediction function
    return NetworkOutput(0, 0, {}, [])

  def get_weights(self):
    # Returns the weights of this network.
    return []

  def training_steps(self) -> int:
    # How many steps / batches the network has been trained for.
    return 0


