import argparse
import os
from datetime import datetime
import json

import torch
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
from utils.misc import setup_process, Logger, save_training_data, load_training_data, cleanup_process

import pickle
import random
import utils.loggers as lg
from config import config
from games.connect4 import Game, GameState
from utils.memory import Memory
from model.model import Residual_CNN
from model.agent import Agent
from utils.funcs import playMatches


def main(rank, args):
    # ============ logging, initialization and directories ==============
    lg.logger_main.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')
    lg.logger_main.info('=*=*=*=*=*=.      NEW LOG      =*=*=*=*=*')
    lg.logger_main.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')
    
    env = Game()
    checkpoint_dir = os.path.join('./run', env.name)
    lg.logger_main.info('Set checkpoint to %s', checkpoint_dir)
    
    memory = Memory(config.MEMORY_SIZE)
    current_NN = Residual_CNN(config.LEARNING_RATE, (2,)+env.grid_shape, env.action_size, config.HIDDEN_CNN_LAYERS)
    best_NN = Residual_CNN(config.LEARNING_RATE, (2,)+env.grid_shape, env.action_size, config.HIDDEN_CNN_LAYERS)
    
    best_player_version = 0
    torch.save(current_NN.state_dict(), os.path.join(checkpoint_dir, 'models', 'best_NN_'+str(0).zfill(4)+'.pt'))
    best_NN.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'models', 'best_NN_'+str(0).zfill(4)+'.pt')))

    current_player = Agent('current_player', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, current_NN)
    best_player = Agent('best_player', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, best_NN)
    iteration = 0
    
    while 1:
        iteration += 1
        print('ITERATION NUMBER', iteration)
        
        lg.logger_main.info('BEST PLAYER VERSION: %d', best_player_version)
        print('BEST PLAYER VERSION', best_player_version)
        
        print('SELF PLAYING', config.EPISODES, 'EPISODES...')
        _, memory, _, _ = playMatches(best_player, best_player, config.EPISODES, lg.logger_main, config.TURNS_UNTIL_TAU0, memory)
        print('\n')
        
        memory.clear_stmemory()
        
        if len(memory.ltmemory) >= config.MEMORY_SIZE:
            print('RETRAINING...')
            current_player.replay(memory.ltmemory)
            print('')
            
            if iteration % 5 == 0:
                pickle.dump(memory, open(os.path.join(checkpoint_dir, 'memory', 'memory_'+str(iteration).zfill(4)+'.pkl'), 'wb'))
                
            lg.logger_memory.info('====================')
            lg.logger_memory.info('NEW MEMORIES')
            lg.logger_memory.info('====================')
            
            memory_samp = random.sample(memory.ltmemory, min(1000, len(memory.ltmemory)))
            for s in memory_samp:
                current_value, current_probs, _ = current_player.get_preds(s['state'])
                best_value, best_probs, _ = best_player.get_preds(s['state'])
                
                lg.logger_memory.info('MCTS VALUE FOR %s: %f', s['playerTurn'], s['value'])
                lg.logger_memory.info('CUR PRED VALUE FOR %s: %f', s['playerTurn'], current_value)
                lg.logger_memory.info('BES PRED VALUE FOR %s: %f', s['playerTurn'], best_value)
                lg.logger_memory.info('THE MCTS ACTION VALUES: %s', ['%.2f' % elem for elem in s['AV']])
                lg.logger_memory.info('CUR PRED ACTION VALUES: %s', ['%.2f' % elem for elem in current_probs])
                lg.logger_memory.info('BES PRED ACTION VALUES: %s', ['%.2f' % elem for elem in best_probs])
                lg.logger_memory.info('ID: %s', s['state'].id)
                lg.logger_memory.info('INPUT TO MODEL: %s', current_player.model.convertToModelInput(s['state']))

                s['state'].render(lg.logger_memory)
            
            print('TOURNAMENT...')
            scores, _, points, sp_scores = playMatches(best_player, current_player, config.EVAL_EPISODES, lg.logger_tourney, 0, None)
            print('\nSCORES')
            print(scores)
            print('\nSTARTING PLAYER / NON-STARTING PLAYER SCORES')
            print(sp_scores)
            print('\n\n')
            
            if scores['current_player'] > scores['best_player'] * config.SCORING_THRESHOLD:
                best_player_version = best_player_version + 1
                torch.save(current_NN.state_dict(), os.path.join(checkpoint_dir, 'models', 'best_NN_' + str(best_player_version).zfill(4) + '.pt'))
                best_NN.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'models', 'best_NN_' + str(best_player_version).zfill(4) + '.pt')))
        else:
            print('MEMORY SIZE:', len(memory.ltmemory))
