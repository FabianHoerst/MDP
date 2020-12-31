import numpy as np

class MDP():
    '''
    Base class for defining an MDP

    Definition of an MDP:
    A Markov Decision Process (MDP) is defined as a 4-tuple <S, A, P, R> where
    - S is a (finite) set of states
    - A is a (finite) set of actions
    - P:SxAxS -> R describes the transition probabilities
    - R:SxAxS -> R describes the reward
    '''

    def __init__(self, states, actions, probabilities, rewards):
        '''
        Initialize the MDP
        :param states: list of states, e.g. states = ('s1', 's2', ...)
        :param actions: list of actions, e.g. actions = ('a1', 'a2', ...)
        :param probabilities: dictionary containing the transition probabilities,
                            structure: P(s,a,s') with s = recent state, a = action to take and s' = successor state
                            e.g. probabilities = {('s', 'a', 's'') : p, ...}
        :param rewards: dictionary containing the reward when beeing in state s, taking action a, and going to state s'
                            structure: R(s,a,s') with s = recent state, a = action to take and s' = successor state
                            e.g. rewards = {('s', 'a', 's'') : r, ...}
        '''
        self.states = states
        self.actions = actions
        self.probabilities = probabilities
        self.rewards = rewards

    def get_states(self):
        '''
        Get the list of defined states
        :return: state list
        '''
        return self.states

    def get_actions(self):
        '''
        Get the list of defined actions
        :return: action list
        '''
        return self.actions

    def get_probability(self, s, action, s_successor):
        '''
        P is used to describe the transition probability P(s_successor|s, action)
        :param s_successor: successor state
        :param action: action to take
        :param s: recent state
        :return: return the transition probability P(s_successor|s, action)
        '''

        return self.probabilities[(s, action, s_successor)]

    def get_reward(self, s, action, s_successor):
        '''
        R is used to return the corresponding reward R(s_successor, action, s)
        :param s_successor: successor state
        :param action: action to take
        :param s: recent state
        :return: return the corresponding reward R(s_successor, action, s)
        '''

        return self.rewards[(s, action, s_successor)]

    # Help classes that are necessary because most of the time MDPs are not fully defined
    def applicable_actions(self, s):
        '''
        Not in every state every action can be taken. Therefore it is necessary to know which action can be taken when
        beeing in a state s
        :param s: state of the MDP
        :return: a list of actions that can be performed when in state
        '''
        actions = []
        # for every possible state
        for state in self.get_states():
            # perform every action
            for action in self.get_actions():
                # test if arc is defined
                if (s, action, state) in self.probabilities:
                    # if action is not in list append it
                    if action not in actions:
                        actions.append(action)

        return actions

    def successor_states(self, s, a):
        '''
        When beeing in state s and taking action a, not every state in S can be reached
        :param s: state
        :param a: action to take
        :return: list of successor states
        '''
        successors = []
        for state in self.states:
            if (s, a, state) in self.probabilities:
                successors.append(state)

        return successors

    def build_P(self, policy):
        '''
        For a given policy (mapping from state to action), calculate the probability transition matrix P(s'|s)
        similar to the P Matrix in Markov Chains
        :param policy: Dictionary mapping from state to action, e.g. policy = { 's1': 'a1', 's2': 'a2',...}
        :return: P_matrix: Matrix containing the transition probabilities from state s to state s',
            e.g. P = [ 1, 0.4, 0,
                       0, 0.2, 0.2,
                       0, 0.4, 0.8 ]
        '''
        # Define Matrix with 0 entries
        p_matrix = np.zeros([len(self.states), len(self.states)])

        # For every state
        for i in range(len(self.states)):
            state = self.states[i]
            # return the action that the policy determines for the given state
            action = policy[state]
            # go over every successor state and get the transition probability
            for j in range(len(self.states)):
                if self.states[j] in self.successor_states(state, action):
                    p_matrix[j, i] = self.get_probability(state, action, self.states[j])
        # Return of matrix
        return p_matrix

    def build_R(self, policy):
        '''
        For a given policy (mapping from state to action), calculate the Reward of every state
        :param policy: Dictionary mapping from state to action, e.g. policy = { 's1': 'a1', 's2': 'a2',...}
        :return: r_matrix: Vector containing the values of the reward function for the states
        '''
        # Define Matrix with 0 entries
        r_matrix = np.zeros(len(self.states))

        # For every state
        for i in range(len(self.states)):
            state = self.states[i]
            # return the action that the policy determines for the given state
            action = policy[state]
            # calculate the associated reward
            for successor in self.successor_states(state, action):
                r_matrix[i] = r_matrix[i] + \
                              self.get_probability(state, action, successor)*self.get_reward(state, action, successor)

        # Return reward matrix for every state
        return r_matrix

    def bellman_eq_policy(self, policy, gamma):
        '''
        calculate the value function for a given policy using bellman equation
        :param policy: Dictionary mapping from state to action
        :param gamma: discount factor gamma, usually gamma = [0,1]
        :return value_function: dictionary containing the value function, e.g. v = { 's1': 123.6, 's2': -1.1,...}
        '''

        # calculate p and r
        p_matrix = self.build_P(policy)
        r_matrix = self.build_R(policy)
        # use bellman equation
        v = r_matrix@np.linalg.inv(np.eye(len(p_matrix))-gamma*p_matrix)
        # build dictionary
        value_function = {}
        states = self.get_states()
        for i in range(len(states)):
            value_function[states[i]] = v[i]

        return value_function

    def value_iteration(self, gamma, max_iterations = 10000, epsilon = 1e-7):
        '''
        Perform the value iteration algorithm to find the best policy for a given state
        :param gamma: discount factor gamma, usually gamma = [0,1]
        :param max_iterations: maximum number of iterations for the value_iteration algorithm
        :param epsilon: defining the error, if the change of the value function in one iteration step is lower than
                        epsilon, the algorithm will terminate
        :return: policy: dictionary containing the best policy, e.g. policy = { 's1': 'a1', 's2': 'a2',...}
        :return: v: dictionary containing the value function of the best policy, e.g. v = { 's1': 123.6, 's2': -1.1,...}
        '''
        # Initialization: v_0 = dict with zeros
        v_n = {state: 0 for state in self.states}
        # iterative algorithm that is running until max_iterations is reached or error is small enough
        for i in range(max_iterations):
            # next value function initialized to zero
            v_n_1 = {}
            # calulate v_n_1 for every state in the MDP
            for state in self.states:
                action_values = {}
                # find action that maximizes the value function
                for action in self.applicable_actions(state):
                    action_values[action] = 0
                    for successor_state in self.successor_states(state, action):
                        value = self.get_probability(state, action, successor_state)*(self.get_reward(state, action, successor_state)+gamma*v_n[successor_state])
                        action_values[action] = action_values[action] + value
                best_action = max(action_values, key=lambda k: action_values[k])
                v_n_1[state] = action_values[best_action]

            # test if the change in value function is small enough
            diff = {state: abs(v_n[state]-v_n_1[state]) for state in self.states}
            max_diff = diff[max(diff, key=lambda k: diff[k])]

            # update v_n
            v_n = v_n_1
            # termination statement
            if max_diff < ((epsilon*(1-gamma))/(2*gamma)):
                break
        # calculate the optimal policy out of the optimal value function
        policy = {}
        for state in self.states:
            action_values = {}
            for action in self.applicable_actions(state):
                action_values[action] = 0
                for successor_state in self.successor_states(state, action):
                    value = self.get_probability(state, action, successor_state) * (
                                self.get_reward(state, action, successor_state) + gamma * v_n[successor_state])
                    action_values[action] = action_values[action] + value
            best_action = max(action_values, key=lambda k: action_values[k])
            # build dictionary
            policy[state] = best_action

        v = v_n
        return policy, v
