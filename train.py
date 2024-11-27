from utils import *
from game import MaxCover
from model import PolicyValueGCN
from mcts_maxcover import MCTS

from greedy import *


import os
import numpy as np
from random import shuffle

import torch
import torch.optim as optim



class Trainer:

    def __init__(self, 
                game,
                model, 
                args):
        self.game = game
        self.model = model
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.mcts = MCTS(self.game, self.model, self.args)

    def exceute_episode(self):

        train_examples = []
        # current_player = 1
        # state = self.game.get_init_board()
        ####
        # graph = nx.barabasi_albert_graph(n=10,m=4)
        # graph = nx.erdos_renyi_graph(n=100,p=0.1)
        # budget = 5
        # game = MaxCover(graph=graph,budget=budget)
        
        state=self.game.get_init_state()

        ###

        while True:
            # print(state)
            # if (graph.number_of_nodes() -np.sum(state))>budget:
            #     raise ValueError('Budget constraint violated')
            # canonical_board = self.game.get_canonical_board(state, current_player)

            # self.mcts = MCTS(self.game, self.model, self.args)
            self.mcts = MCTS(game=self.game,model=self.model,args=self.args)
            root = self.mcts.run(model=self.model,state=state)
 
            action_probs = [0 for _ in range(self.game.get_action_size())]
            # action_probs = [0 ]
            for k, v in root.children.items():
                action_probs[k] = v.visit_count

            action_probs = action_probs / np.sum(action_probs)
            train_examples.append((state,action_probs))

            action = root.select_action(temperature=0)
            # state = self.game.get_next_state(state, action)
            state = self.game.get_next_state(state=state,action=action)
            # state, current_player = self.game.get_next_state(state, current_player, action)
            # reward = self.game.get_reward_for_player(state, current_player)

            # reward = self.game.get_reward_for_player(state)
            reward = self.game.get_reward_for_player(state)

            if reward is not None:
                ret = []

                # data = from_networkx(self.game.graph)
                data = from_networkx(self.game.graph)
                # data.x = torch.from_numpy(next_state)
                # data = Batch.from_data_list([data])
                for hist_state,hist_action_probs in train_examples:
                    # data_copy = data.copy()
                    data_copy = data.clone()  # Use clone for deep copy instead of copy()
                    data_copy.x = torch.from_numpy(hist_state)
                    hist_action_probs[self.game.action_mask] *=0.5/len(self.game.action_mask)
                    hist_action_probs[self.game.action_demask] *= 0.5 / len(self.game.action_demask)
                    ret.append((data_copy,hist_action_probs,reward))
                    # [Board, currentPlayer, actionProbabilities, Reward]
                    # ret.append((hist_state, hist_action_probs, reward * ((-1) ** (hist_current_player != current_player))))

                return ret

    def learn(self):
        for i in range(1, self.args['numIters'] + 1):

            print("{}/{}".format(i, self.args['numIters']))

            train_examples = []

            for eps in range(self.args['numEps']):
                iteration_train_examples = self.exceute_episode()
                train_examples.extend(iteration_train_examples)

            shuffle(train_examples)
            self.train(train_examples)
            filename = self.args['checkpoint_path']
            self.save_checkpoint(folder=".", filename=filename)

    def train(self, examples):
        optimizer = optim.Adam(self.model.parameters(), lr=5e-4)
        pi_losses = []
        v_losses = []

        print(len(examples))

        for epoch in range(self.args['epochs']):
            self.model.train()

            batch_idx = 0

            while batch_idx < int(len(examples) / self.args['batch_size']):
                sample_ids = np.random.randint(len(examples), size=self.args['batch_size'])
                # boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                graphs, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                data = Batch.from_data_list(graphs).to(self.device)
                # boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis)).reshape(-1,1).to(self.device)
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64)).to(self.device)

                # print(torch.sum(target_pis[:100]))
                # print(target_vs.shape)

                pred_pis,pred_vs = self.model(data)

                  # Policy loss: KL divergence since both pred_pis and target_pis are probabilities
                # pi_loss = torch.nn.functional.kl_div(pred_pis.log(), target_pis, reduction='batchmean')

                pi_loss = self.loss_pi(target_pis, pred_pis)
                v_loss = self.loss_v(target_vs,pred_vs)

                # # Value loss: MSE for value regression
                # v_loss = torch.nn.functional.mse_loss(pred_vs, target_vs)

                total_loss = pi_loss + v_loss

                # Backward pass and optimizer step
                optimizer.zero_grad()  # Clear gradients
                total_loss.backward()  # Backpropagate
                optimizer.step()  # Update weights

                # Record the losses for later analysis
                pi_losses.append(pi_loss.item())
                v_losses.append(v_loss.item())

                batch_idx += 1

        # print(pi_losses)
        # print(v_losses)  

    def loss_pi(self, targets, outputs):
        loss = -(targets * torch.log(outputs)).sum(dim=1)
        return loss.mean()

    def loss_v(self, targets, outputs):
        loss = torch.sum((targets-outputs.view(-1))**2)/targets.size()[0]
        return loss

    def save_checkpoint(self, folder, filename):
        if not os.path.exists(folder):
            os.mkdir(folder)

        filepath = os.path.join(folder, filename)
        torch.save(self.model.state_dict(),filepath )
