import torch
import models
import pickle
import time
from game import Game, Action
from mcts import Node, run_mcts, select_action, find_subtree_root
from network import NetworkOutput
from config import MuZeroConfig_TicTacToe

def expand_root(config, game, network, root):
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
    #  root.add_exploration_noise(config)

def play(config, game, network, player, reuse_tree, last_action, verbose=False):
  if player=='reuse' and reuse_tree:
    root = find_subtree_root(reuse_tree, last_action)
    if root.expanded():
      #  root.add_exploration_noise(config)
      pass
    else:
      expand_root(config, game, network, root)
  else:
    # At the root of the search tree we use the representation function to obtain a hidden state given the current observation.
    root = Node(0)
    expand_root(config, game, network, root)

  # We then run a Monte Carlo Tree Search using only action sequences and the model learned by the network.
  run_mcts(config, root, game.action_history(), network)
  action = select_action(config, len(game.history), root, network)
  done = game.apply(action.index)

  subtree_root = find_subtree_root(root, action)

  return subtree_root, action, done

def run_play_against(config, nn, against_len, player0='reuse', player1='default', dump_game=False, verbose=False):
  p0_win = 0
  p1_win = 0
  tie = 0
  p0_time = 0.0
  p1_time = 0.0
  
  moves_first = 0
  for i in range(against_len):
    game = config.new_game()
    current_player = moves_first
    player0_tree = None 
    player1_tree = None 
    last_action = None
    done = False
    #  print(moves_first, 'moves first!')
    while not done and len(game.history) < config.max_moves:
      start = time.time()
      if current_player == 0:
        player0_tree, action, done = play(config, game, nn, player0, player0_tree, last_action)
        p0_time += (time.time()-start)
      else:
        player1_tree, action, done = play(config, game, nn, player1, player1_tree, last_action)
        p1_time += (time.time()-start)
      last_action = action
      current_player = 1 - current_player

    if game.terminal(): 
      if current_player == 0:
        p1_win += 1
        #  print('p1 wins')
      else:
        p0_win += 1
        #  print('p0 wins')
    else:
      tie += 1
      #  print('tie')

    #  print('======================')
    
    moves_first = 1 - moves_first

  print('p0 average step time is', p0_time/against_len)
  print('p1 average step time is', p1_time/against_len)

  return p0_win, p1_win, tie

if __name__ == "__main__":
  config = MuZeroConfig_TicTacToe('resnet')
  ckp = torch.load('nn_model/tictactoe/resnet.checkpoint')

  nn = models.MuZeroNetwork(config)
  nn.eval()
  nn.set_weights(ckp['weights'])

  p0_win, p1_win, tie = run_play_against(config, nn, 100, player0='reuse', player1='default')

  print('p0 wins', p0_win)
  print('p1 wins', p1_win)
  print('tie', tie)
