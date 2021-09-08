import torch
import util
import models
import pickle
from game import Game, Action
from mcts import Node, run_mcts, select_action, find_subtree_root
from network import NetworkOutput
from config import MuZeroConfig_TicTacToe, MuZeroConfig_connect4


def play_game(config, network, dump_game=False, reuse_subtree=False, verbose=False):
  game = config.new_game()
  steps = []
  first_step = True
  while not game.terminal() and len(game.history) < config.max_moves:
    if reuse_subtree and first_step==False:
      root = subtree_root
    else:
      # At the root of the search tree we use the representation function to obtain a hidden state given the current observation.
      root = Node(0)
      current_observation = game.make_image(-1)
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

    # We then run a Monte Carlo Tree Search using only action sequences and the model learned by the network.
    run_mcts(config, root, game.action_history(), network)
    action = select_action(config, len(game.history), root, network)
    game.apply(action.index)
    game.store_search_statistics(root)
    first_step = False

    if verbose:
      print(game.environment.board, game.environment.to_play(), action.index)

    if reuse_subtree:
      subtree_root = find_subtree_root(root, action)

    if dump_game:
      steps.append([root, action])
  
  if dump_game:
    with open(dump_game, 'wb') as outp:
      pickle.dump(steps, outp, pickle.HIGHEST_PROTOCOL)

  return game


def run_selfplay(config, nn_storage, replay_buffer, selfplay_len, dump_game=False, reuse_subtree=False, verbose=False):
  nn = models.MuZeroNetwork(config)
  nn.eval()
  for i in range(selfplay_len):
    nn.set_weights(nn_storage.latest_network())
    game = play_game(config, nn, dump_game, reuse_subtree, verbose)
    replay_buffer.save_game(game)

if __name__ == "__main__":
  #  config = MuZeroConfig_TicTacToe('resnet')
  config = MuZeroConfig_connect4('resnet')
  nn_storage = util.SharedStorage()
  replay_buffer = util.ReplayBuffer(config)

  # Put one NN in the storage
  nn_storage.networks[0] = torch.load('nn_model/connect4/resnet.checkpoint')

  run_selfplay(config, nn_storage, replay_buffer, 1, dump_game='connect4.pkl',verbose=True)
