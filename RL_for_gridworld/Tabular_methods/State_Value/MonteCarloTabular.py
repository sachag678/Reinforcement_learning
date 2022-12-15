"""Sacha Gunaratne | github@sachag678"""
from Tabular_methods.State_Value.StateValueTabular import StateValueTabular

class MonteCarloTabular(StateValueTabular):
    """Learns optimum behavior using the iterative tabular MC method."""
    
    def batch_update(self, states, rewards, actions):
        """Update the model using the states, rewards using the iterative MC update.
            V(s_t) = V(s_t) + alpha(G_t - V(s_t))
        """
        states = states[1:]  # handle the fact that we are doing both state-value and state-action
        discounted_rewards = self.discount_reward(rewards)
        for state, reward in zip(states, discounted_rewards):
            self.state_values[self.convert_to_immutable(state)] = self.state_values[self.convert_to_immutable(state)] + self.alpha * (reward - self.get_value(state))
    
    
    def discount_reward(self, rewards):
        """Calculate the future discounted reward."""
        discounted_rewards = []
        for _ in range(len(rewards)):
            temp_r = 0
            for index, r in enumerate(rewards):
                temp_r += (self.gamma**index)*r
            discounted_rewards.append(temp_r)
        return discounted_rewards
    


