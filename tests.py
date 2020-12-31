import unittest

from mdp import MDP
import numpy as np
import numpy.testing as npt

# define Test MDP1
states1 = ('s1', 's2')
actions1 = ('a1', 'a2')
probabilities1 = {
    ('s1', 'a1', 's1'): 0.9,
    ('s1', 'a1', 's2'): 0.1,
    ('s2', 'a2', 's1'): 1.0
}
rewards1 = {
    ('s1', 'a1', 's1'): 20.0,
    ('s1', 'a1', 's2'): 0.0,
    ('s2', 'a2', 's1'): 0.0
}

# define Test MDP2
states2 = ('a', 'b', 'c')
actions2 = ('1', '2')
probabilities2 = {
    ('a', '1', 'b'): 1.0,
    ('a', '2', 'c'): 1.0,

    ('b', '1', 'a'): 0.5,
    ('b', '1', 'c'): 0.5,
    ('b', '2', 'a'): 0.5,
    ('b', '2', 'c'): 0.5,

    ('c', '1', 'a'): 0.2,
    ('c', '1', 'b'): 0.8,
    ('c', '2', 'a'): 0.7,
    ('c', '2', 'b'): 0.3,
}
rewards2 = {
    ('a', '1', 'b'): 0,
    ('a', '2', 'c'): 0,

    ('b', '1', 'a'): 1,
    ('b', '1', 'c'): 1,
    ('b', '2', 'a'): 1,
    ('b', '2', 'c'): 1,

    ('c', '1', 'a'): 2,
    ('c', '1', 'b'): 2,
    ('c', '2', 'a'): 2,
    ('c', '2', 'b'): 2,
}

class TestMDP(unittest.TestCase):

    def test_S_method(self):
        '''
        Test if the get_states method of the class MDP is returning the state list of the defined MDP
        '''
        # build MDPs
        mdp1 = MDP(states1, actions1, probabilities1, rewards1)
        mdp2 = MDP(states2, actions2, probabilities2, rewards2)
        # Tests
        self.assertEqual(mdp1.get_states(), states1, 'get_states Method test failed')
        self.assertEqual(mdp2.get_states(), states2, 'get_states Method test failed')

    def test_A_method(self):
        '''
        Test if the get_actions method of the class MDP is returning the action list of the defined MDP
        '''
        # build MDPs
        mdp1 = MDP(states1, actions1, probabilities1, rewards1)
        mdp2 = MDP(states2, actions2, probabilities2, rewards2)
        # Tests
        self.assertEqual(mdp1.get_actions(), actions1, 'get_actions Method test failed')
        self.assertEqual(mdp2.get_actions(), actions2, 'get_actions Method test failed')

    def test_P_method(self):
        '''
        Test if the get_probability method of the class MDP is returning the correct probability
        '''
        # build MDPs
        mdp1 = MDP(states1, actions1, probabilities1, rewards1)
        mdp2 = MDP(states2, actions2, probabilities2, rewards2)
        # Tests
        p_return1 = {element: mdp1.get_probability(element[0], element[1], element[2]) for element in probabilities1}
        p_return2 = {element: mdp2.get_probability(element[0], element[1], element[2]) for element in probabilities2}
        self.assertEqual(p_return1, probabilities1, 'get_probability Method test failed')
        self.assertEqual(p_return2, probabilities2, 'get_probability Method test failed')

    def test_R_method(self):
        '''
        Test if the get_reward method of the class MDP is returning the correct probability
        '''
        # build MDPs
        mdp1 = MDP(states1, actions1, probabilities1, rewards1)
        mdp2 = MDP(states2, actions2, probabilities2, rewards2)
        # Tests
        r_return1 = {element: mdp1.get_reward(element[0], element[1], element[2]) for element in probabilities1}
        r_return2 = {element: mdp2.get_reward(element[0], element[1], element[2]) for element in probabilities2}
        self.assertEqual(r_return1, rewards1, 'get_reward Method test failed')
        self.assertEqual(r_return2, rewards2, 'get_reward Method test failed')

    def test_applicable_actions_method(self):
        '''
        Test if the applicable_actions method of the class MDP is returning the correct actions that can be taken when
        beeing in a state s
        '''
        # build MDPs
        mdp1 = MDP(states1, actions1, probabilities1, rewards1)
        mdp2 = MDP(states2, actions2, probabilities2, rewards2)
        self.assertEqual(mdp1.applicable_actions('s1'), ['a1'], 'applicable_actions Method test failed')
        self.assertEqual(mdp2.applicable_actions('a'), ['1', '2'], 'applicable_actions Method test failed')

    def test_successor_states_method(self):
        '''
        Test if the successor_states method of the class MDP is returning the possible successor states when
        beeing in a state s and take action a
        '''
        mdp1 = MDP(states1, actions1, probabilities1, rewards1)
        mdp2 = MDP(states2, actions2, probabilities2, rewards2)
        self.assertEqual(mdp1.successor_states('s1', 'a1'), ['s1', 's2'], 'successor_states Method test failed')
        self.assertEqual(mdp2.successor_states('b', '1'), ['a', 'c'], 'successor_states Method test failed')

    def test_build_P(self):
        '''
        Test if the build_P Method is returning the correct Probability matrix for a given policy
        '''
        mdp1 = MDP(states1, actions1, probabilities1, rewards1)
        p1 = np.array([[0.9, 1], [0.1, 0]])
        policy1 = {
            's1': 'a1',
            's2': 'a2'
        }
        mdp2 = MDP(states2, actions2, probabilities2, rewards2)
        p2 = np.array([[0, 0.5, 0.2],[1, 0, 0.8], [0, 0.5, 0]])
        policy2 = {
            'a': '1',
            'b': '1',
            'c': '1'
        }
        npt.assert_array_equal(mdp1.build_P(policy1), p1, 'build_P Method test failed')
        npt.assert_array_equal(mdp2.build_P(policy2), p2, 'build_P Method test failed')

    def test_build_R(self):
        '''
        Test if the build_R Method is returning the correct return for a state
        '''
        r = np.array([18, 0])
        policy = {
            's1': 'a1',
            's2': 'a2'
        }
        mdp = MDP(states1, actions1, probabilities1, rewards1)
        npt.assert_array_equal(mdp.build_R(policy), r, 'build_R Method test failed')

    def test_bellman_eq_policy(self):
        '''
        Test if the bellman_eq_policy Method is returning the correct value functions for a given policy
        '''
        mdp1 = MDP(states1, actions1, probabilities1, rewards1)
        policy1 = {
            's1': 'a1',
            's2': 'a2'
        }
        value_function11 = np.array([34.285714, 17.142857])
        value_function12 = np.array([1637.852593, 1621.474067])

        mdp2 = MDP(states2, actions2, probabilities2, rewards2)
        policy2 = {
            'a': '1',
            'b': '1',
            'c': '1'
        }
        value_function21 = np.array([8.63, 9.59, 10.46])

        npt.assert_array_almost_equal(mdp1.bellman_eq_policy(policy1, 0.5), value_function11, 6, \
                                      'bellman_eq_policy Method test failed')
        npt.assert_array_almost_equal(mdp1.bellman_eq_policy(policy1, 0.99), value_function12, 6, \
                                      'bellman_eq_policy Method test failed')
        npt.assert_array_almost_equal(mdp2.bellman_eq_policy(policy2, 0.9), value_function21, 2, \
                                      'bellman_eq_policy Method test failed')

    def test_value_iteration(self):
        '''
        Test if the value_iteration algorithm is returning the correct optimal policy and value function
        '''
        mdp1 = MDP(states1, actions1, probabilities1, rewards1)
        optimal_policy1 = {'s1':'a1', 's2':'a2'}
        policy1, value_function1 = mdp1.value_iteration(0.5, 10000, 1e-7)
        mdp2 = MDP(states2, actions2, probabilities2, rewards2)
        optimal_policy2 = {'a': '2', 'b': '1', 'c': '1'}
        policy2, value_function2 = mdp2.value_iteration(0.9, 10000, 1e-7)

        self.assertEqual(policy1, optimal_policy1, 'bellman_eq_policy Method test failed')
        self.assertEqual(value_function1, {'s1': 34.285714255234645, 's2': 17.142857112377495}, \
                         'bellman_eq_policy Method test failed')
        self.assertEqual(policy2, optimal_policy2, 'bellman_eq_policy Method test failed')
        self.assertEqual(value_function2, {'a': 11.007194195249909, 'b': 11.45683448301969, 'c': 12.23021577798372}, \
                         'bellman_eq_policy Method test failed')


if __name__ == '__main__':
    #unittest.main()
    help(MDP)
