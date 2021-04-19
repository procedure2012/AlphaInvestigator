import random
import numpy as np

from config import config
import MCTS as mc
import utils.loggers as lg


class User():
    def __init__(self, name, state_size, acton_size):
        self.name = name
        self.state_size = state_size
        self.action_size = acton_size
 
    def act(self, state, tau):
        action = input('Enter your chosen action: ')
        pi = np.zeros(self.action_size)
        pi[action] = 1
        value = None
        NN_value = None
        return action, pi, value, NN_value

class Agent():
    def __init__(self, name, state_size, action_size, mcts_simulations, cpuct, model):
        self.name = name
        self.state_size = state_size
        self.action_szie = action_size
        self.cpuct = cpuct
        self.MCTSsimulations = mcts_simulations
        self.model = model
        self.mcts = None
        self.train_overall_loss = []
        self.train_value_loss = []
        self.train_policy_loss = []
        self.val_overall_loss = []
        self.val_value_loss = []
        self.val_policy_loss = []
        
    def simulate(self):
        lg.logger_mcts.info('ROOT NODE...%s', self.mcts.root.state.id)
        self.mcts.root.state.render(lg.logger_mcts)
        lg.logger_mcts.info('CURRENT PLAYER...%s', self.mcts.root.state.playerTurn)
        
        leaf, value, done, breadcrumbs = self.mcts.moveToLeaf()
        leaf.state.render(lg.logger_mcts)
        
        value, breadcrumbs = self.evaluateLeaf(leaf, value, done, breadcrumbs)
        
        self.mcts.backFill(leaf, value, breadcrumbs)
    
    def act(self, state, tau):
        if self.mcts is None or state.id not in self.mcts.tree:
            self.buildMCTS(state)
        else:
            self.changeRootMCTS(state)
        
        for sim in range(self.MCTSsimulations):
            lg.logger_mcts.info('***************************')
            lg.logger_mcts.info('****** SIMULATION %d ******', sim+1)
            lg.logger_mcts.info('***************************')
            self.simulate()
            
        pi, values = self.getAV(1)
        action, value = self.chooseAction(pi, values, tau)
        nextState, _, _ = state.takeAction(action)
        NN_value = -self.get_preds(nextState)[0]
        
        lg.logger_mcts.info('ACTION VALUES...%s', pi)
        lg.logger_mcts.info('CHOSEN ACTION...%d', action)
        lg.logger_mcts.info('MCTS PERCEIVED VALUE...%f', value)
        lg.logger_mcts.info('NN PERCEIVED VALUE...%f', NN_value)

        return action, pi, value, NN_value
    
    def get_preds(self, state):
        inputToModel = np.array([self.model.covertToModeInput(state)])
        preds = self.model(inputToModel)
        
        value_array = preds[0]
        logits_array = preds[1]
        value = value_array[0]
        logits = logits_array[0]
        allowedActions = state.allowedActions
        
        mask = np.ones(logits.shape, dtype=bool)
        mask[allowedActions] = False
        logits[mask] = -100
        
        odds = np.exp(logits)
        probs = odds / np.sum(odds)
        return value, probs, allowedActions
    
    def evaluateLeaf(self, leaf, value, done, breadcrumbs):
        lg.logger_mcts.info('------EVALUATING LEAF------')
        
        if done == 0:
            value, probs, allowedActions = self.get_preds(leaf.state)
            lg.logger_mcts.info('PREDICTED VALUE FOR %d: %f', leaf.state.playerTurn, value)
            
            probs = probs[allowedActions]
            
            for idx, action in enumerate(allowedActions):
                newState, _, _ = leaf.takeAction(action)
                if newState.id not in self.mcts.tree:
                    node = mc.Node(newState)
                    self.mcts.addNode(node)
                    lg.logger_mcts.info('added node node...%s...p = %f', node.id, probs[idx])
                else:
                    node = self.mcts.tree[newState.id]
                    lg.logger_mcts.info('existing node...%s...', node.id)
                    
                newEdge = mc.Edge(leaf, node, probs[idx], action)
                leaf.edges.append((action, newEdge))
        else:
            lg.logger_mcts.info('GAME VALUE FOR %d: %f', leaf.playerTruen, value)
            
        return value, breadcrumbs

    def getAV(self, tau):
        edges = self.mcts.root.edges
        pi = np.zeros(self.action_size, dtype=np.integer)
        values = np.zeros(self.action_size, dtype=np.float32)
        
        for action, edge in edges:
            pi[action] = pow(edge.stats['N'], 1/tau)
            values[action] = edges.stats['Q']
        
        pi = pi / (np.sum(pi) * 1.0)
        return pi, values
    
    def chooseAction(self, pi, values, tau):
        if tau == 0:
            actions = np.argwhere(pi == max(pi))
            action = random.choice(actions)[0]
        else:
            action_idx = np.random.multinomial(1, pi)
            action = np.where(action_idx == 1)[0][0]
            
        value = values[action]
        
        return action, value
    
    def replay(self, ltmemory):
        lg.logger_mcts.info('******RETRAINING MODEL******')
        
        for i in range(config.TRAINING_LOOPS):
            minibatch = random.sample(ltmemory, min(config.BATCH_SIZE, len(ltmemory)))
            training_states = np.array([self.model.convertToModelInput(row['state']) for row in minibatch])
            training_targets = {'value_head': np.array([row['value'] for row in minibatch]),
                                'policy_head': np.array([row['AV'] for row in minibatch])}
            fit = self.model(training_states, training_targets)
            lg.logger_mcts.info('NEW LOSS %s', fit)
            # TODO
            # Add backward
            self.train_overall_loss.append(round(fit['loss'][config.EPOCHS - 1], 4))
            self.train_value_loss.append(round(fit['value_head_loss'][config.EPOCHS - 1], 4))
            self.train_policy_loss.append(round(fit['policy_head_loss'][config.EPOCHS - 1], 4))

    def buildMCTS(self, state):
        lg.logger_mcts.info('****** BUILDING NEW MCTS TREE FOR AGENT %s ******', self.name)
        self.root = mc.Node(state)
        self.mcts = mc.MCTS(self.root, self.cpuct)
    
    def changeRootMCTS(self, state):
        lg.logger_mcts.info('****** CHANGING ROOT OF MCTS TREE TO %s FOR AGENT ******', state.id, self.name)
        self.mcts.root = self.mcts.tree[state.id]