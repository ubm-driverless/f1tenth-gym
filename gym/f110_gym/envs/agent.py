class Agent:
    def __init__(self, agent_obj):
        self.agent_obj = agent_obj

        self.parameters_space = agent_obj.parameters_space

    def reset(self):
        """
        Reset the agent to its initial state.
        """
        self.agent_obj.reset()

    def update_parameters(self, parameters):
        """
        Update the parameters of the agent.

        :param parameters: The new parameters to set.
        :type parameters: np.ndarray
        """

        self.agent_obj.validate_parameters(parameters)

        self.agent_obj.set_parameters(parameters)

    def get_action(self, agent_data):
        """
        Perform a step in the simulation.
        """

        self.agent_obj.validate_agent_data(agent_data)
        self.agent_obj.get_action(agent_data)
