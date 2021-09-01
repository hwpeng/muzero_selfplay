import torch
import util
import models
from game import Game, Action
from mcts import Node, run_mcts, select_action
from network import NetworkOutput
from config import MuZeroConfig_TicTacToe


def play_game(config, network):
  game = config.new_game()
  while not game.terminal() and len(game.history) < config.max_moves:
    # At the root of the search tree we use the representation function to
    # obtain a hidden state given the current observation.
    root = Node(0)
    current_observation = game.make_image(-1)

    #  print(game.environment.board)

    (value, reward, p_logits, hidden_state) = network.initial_inference(current_observation)

    policy_logits = {}
    for i in range(config.action_space_size):
      policy_logits[Action(i)] = p_logits[0][i].item()

    nn_outputs = NetworkOutput(
        models.support_to_scalar(value, config.support_size), 
        models.support_to_scalar(reward, config.support_size), 
        policy_logits,
        hidden_state)

    root.expand_node(game.to_play(), game.legal_actions(), nn_outputs)
    root.add_exploration_noise(config)

    # We then run a Monte Carlo Tree Search using only action sequences and the
    # model learned by the network.
    run_mcts(config, root, game.action_history(), network)
    action = select_action(config, len(game.history), root, network)
    game.apply(action.index)
    game.store_search_statistics(root)
  return game


def run_selfplay(config, nn_storage, replay_buffer, selfplay_len):
  nn = models.MuZeroNetwork(config)
  nn.eval()
  for i in range(selfplay_len):
    #  print('Game', i)
    nn.set_weights(nn_storage.latest_network())
    game = play_game(config, nn)
    replay_buffer.save_game(game)

if __name__ == "__main__":
  #  config = util.make_tictactoe_config()
  config = MuZeroConfig_TicTacToe('fullyconnected')
  nn_storage = util.SharedStorage()
  replay_buffer = util.ReplayBuffer(config)

  # Put one NN in the storage
  nn_storage.networks[0] = torch.load('nn_model/tictactoe/fc.checkpoint')


  run_selfplay(config, nn_storage, replay_buffer, 100)
