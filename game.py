import numpy
import torch

class Environment():
  """The environment MuZero is interacting with."""
  def __init__(self, name):
    self.name = name

  def step(self, action):
    pass

class Player(object):
  pass

class Action(object):

  def __init__(self, index: int):
    self.index = index

  def __hash__(self):
    return self.index

  def __eq__(self, other):
    return self.index == other.index

  def __gt__(self, other):
    return self.index > other.index

class ActionHistory(object):
  """Simple history container used inside the search.

  Only used to keep track of the actions executed.
  """

  def __init__(self, history, action_space_size: int):
    self.history = list(history)
    self.action_space_size = action_space_size

  def clone(self):
    return ActionHistory(self.history, self.action_space_size)

  def add_action(self, action: Action):
    self.history.append(action)

  def last_action(self) -> Action:
    return self.history[-1]

  def action_space(self):
    return [Action(i) for i in range(self.action_space_size)]

  def to_play(self) -> Player:
    return Player()

class Game():
  """A single episode of interaction with the environment."""

  def __init__(self, game_name, action_space_size, discount):
    if game_name == 'tictactoe':
      self.environment = TicTacToe()
    else:
      self.environment = Environment(game_name)  # Game specific environment.
    self.history = []
    self.rewards = []
    self.child_visits = []
    self.root_values = []
    self.action_space_size = action_space_size
    self.discount = discount

  def terminal(self):
    return self.environment.terminal()

  def legal_actions(self):
    # Game specific calculation of legal actions.
    return self.environment.legal_actions()

  def apply(self, action):
    obs, reward, done = self.environment.step(action)
    self.rewards.append(reward)
    self.history.append(action)
    return done

  def store_search_statistics(self, root):
    sum_visits = sum(child.visit_count for child in root.children.values())
    action_space = (Action(index) for index in range(self.action_space_size))
    self.child_visits.append([
        root.children[a].visit_count / sum_visits if a in root.children else 0
        for a in action_space
    ])
    self.root_values.append(root.value())

  def make_image(self, state_index):
    board = self.environment.get_observation()
    return torch.tensor(board).float().unsqueeze(0)

  def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int, to_play):
    # The value target is the discounted root value of the search tree N steps
    # into the future, plus the discounted sum of all rewards until then.
    targets = []
    for current_index in range(state_index, state_index + num_unroll_steps + 1):
      bootstrap_index = current_index + td_steps
      if bootstrap_index < len(self.root_values):
        value = self.root_values[bootstrap_index] * self.discount**td_steps
      else:
        value = 0

      for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
        value += reward * self.discount**i  # pytype: disable=unsupported-operands

      # For simplicity the network always predicts the most recently received
      # reward, even for the initial representation network where we already
      # know this reward.
      if current_index > 0 and current_index <= len(self.rewards):
        last_reward = self.rewards[current_index - 1]
      else:
        last_reward = 0

      if current_index < len(self.root_values):
        targets.append((value, last_reward, self.child_visits[current_index]))
      else:
        # States past the end of games are treated as absorbing states.
        targets.append((0, last_reward, []))
    return targets

  def to_play(self):
    return self.environment.to_play()

  def action_history(self) -> ActionHistory:
    return ActionHistory(self.history, self.action_space_size)

class TicTacToe:
  def __init__(self):
    self.board = numpy.zeros((3, 3), dtype="int32")
    self.player = 1

  def to_play(self):
    return 0 if self.player == 1 else 1

  def reset(self):
    self.board = numpy.zeros((3, 3), dtype="int32")
    self.player = 1
    return self.get_observation()

  def step(self, action):
    row = action // 3
    col = action % 3
    self.board[row, col] = self.player

    done = self.terminal() or len(self.legal_actions()) == 0

    reward = 1 if self.terminal() else 0

    if not done:
      self.player *= -1

    return self.get_observation(), reward, done

  def get_observation(self):
    board_player1 = numpy.where(self.board == 1, 1, 0)
    board_player2 = numpy.where(self.board == -1, 1, 0)
    board_to_play = numpy.full((3, 3), self.player)
    return numpy.array([board_player1, board_player2, board_to_play], dtype="int32")

  def legal_actions(self):
    legal = []
    for i in range(9):
      row = i // 3
      col = i % 3
      if self.board[row, col] == 0:
        legal.append(Action(i))
    return legal

  def terminal(self):
    # Horizontal and vertical checks
    for i in range(3):
      if (self.board[i, :] == self.player * numpy.ones(3, dtype="int32")).all():
        return True
      if (self.board[:, i] == self.player * numpy.ones(3, dtype="int32")).all():
        return True

    # Diagonal checks
    if (
      self.board[0, 0] == self.player
      and self.board[1, 1] == self.player
      and self.board[2, 2] == self.player
    ):
      return True
    if (
      self.board[2, 0] == self.player
      and self.board[1, 1] == self.player
      and self.board[0, 2] == self.player
    ):
      return True

    return False

