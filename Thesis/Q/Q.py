import numpy as np
import random as rand


class QLearner(object):
    def __init__(self, num_states=100, num_actions=4, alpha=0.2, gamma=0.9, rar=0.5, radr=0.99, dyna=0, verbose=False):
        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.s = 0
        self.a = 0
        self.Q = np.zeros((self.num_states, self.num_actions))
        if self.dyna > 0:
            self.T = np.ones((self.num_states, self.num_actions, self.num_states))
            self.R = np.zeros((self.num_states, self.num_actions))
            self.Tc = np.ones((self.num_states, self.num_actions, self.num_states)) / 1000000

    # Update the state without updating, but using the Q-table: s is the new state and a is the returned action
    def querysetstate(self, s):
        self.s = s
        rnum = rand.random()
        if rnum < self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.Q[s, :])
        if self.verbose:
            print "s =", s, "a =", action
        return action

    # Update the Q table: s_prime is the new state and returns the selected action
    def query(self, s_prime, r):
        self.Q[self.s, self.a] = (1 - self.alpha) * self.Q[self.s, self.a] + self.alpha * (r + self.gamma * self.Q[s_prime, np.argmax(self.Q[s_prime, :])])
        if self.dyna > 0:
            alphar = 0.9
            self.Tc[self.s, self.a, s_prime] += 1
            self.T[self.s, self.a, :] = self.Tc[self.s, self.a, :]/(self.Tc[self.s, self.a, :]).sum()
            self.R[self.s, self.a] = (1 - alphar) * self.R[self.s, self.a] + alphar * r
            for i in range(self.dyna):
                rand_s = rand.randint(0, self.num_states - 1)
                rand_a = rand.randint(0, self.num_actions - 1)
                dyna_s_prime = np.where(np.cumsum(self.T[rand_s, rand_a, :]) >= rand.random())[0][0]
                dyna_r = self.R[rand_s, rand_a]
                self.Q[rand_s, rand_a] = (1 - self.alpha) * self.Q[rand_s, rand_a] + self.alpha * (dyna_r + self.gamma * self.Q[dyna_s_prime, np.argmax(self.Q[dyna_s_prime, :])])

        self.s = s_prime
        action = self.querysetstate(self.s)
        self.a = action
        self.rar = self.rar * self.radr
        if self.verbose: print "s' =", s_prime, "a =", action, "r =", r
        return action
