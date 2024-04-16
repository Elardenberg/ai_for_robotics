import random
import math
from constants import *

class FiniteStateMachine(object):
    """
    A finite state machine.
    """
    def __init__(self, state):
        self.state = state

    def change_state(self, new_state):
        self.state = new_state

    def update(self, agent):
        self.state.check_transition(agent, self)
        self.state.execute(agent)


class State(object):
    """
    Abstract state class.
    """
    def __init__(self, state_name):
        """
        Creates a state.

        :param state_name: the name of the state.
        :type state_name: str
        """
        self.state_name = state_name

    def check_transition(self, agent, fsm):
        """
        Checks conditions and execute a state transition if needed.

        :param agent: the agent where this state is being executed on.
        :param fsm: finite state machine associated to this state.
        """
        raise NotImplementedError("This method is abstract and must be implemented in derived classes")

    def execute(self, agent):
        """
        Executes the state logic.

        :param agent: the agent where this state is being executed on.
        """
        raise NotImplementedError("This method is abstract and must be implemented in derived classes")


class MoveForwardState(State):
    def __init__(self):
        # Todo: add initialization code
        super().__init__("MoveForward")
        self.cont = 0
        self.t = self.cont

    def check_transition(self, agent, state_machine):
        # Todo: add logic to check and execute state transition
        if self.t > MOVE_FORWARD_TIME:
            state_machine.change_state(MoveInSpiralState())
        elif agent.get_bumper_state():
            state_machine.change_state(GoBackState())

    def execute(self, agent):
        # Todo: add execution logic
        self.t = self.cont * SAMPLE_TIME
        agent.set_velocity(FORWARD_SPEED, 0)
        self.cont += 1


class MoveInSpiralState(State):
    def __init__(self):
        super().__init__("MoveInSpiral")
        # Todo: add initialization code
        self.cont = 0
        self.t = self.cont
        self.radius = INITIAL_RADIUS_SPIRAL
        self.angular_speed = ANGULAR_SPEED
    
    def check_transition(self, agent, state_machine):
        # Todo: add logic to check and execute state transition
        if self.t > MOVE_IN_SPIRAL_TIME:
            state_machine.change_state(MoveForwardState())
        elif agent.get_bumper_state():
            state_machine.change_state(GoBackState())

    def execute(self, agent):
        self.t = self.cont * SAMPLE_TIME
        self.radius = INITIAL_RADIUS_SPIRAL + SPIRAL_FACTOR * self.t
        self.angular_speed = FORWARD_SPEED / self.radius
        agent.set_velocity(FORWARD_SPEED, self.angular_speed)
        self.cont += 1


class GoBackState(State):
    def __init__(self):
        super().__init__("GoBack")
        # Todo: add initialization code
        self.cont = 0
        self.t = self.cont

    def check_transition(self, agent, state_machine):
        # Todo: add logic to check and execute state transition
        if self.t > GO_BACK_TIME:
            state_machine.change_state(RotateState())

    def execute(self, agent):
        # Todo: add execution logic
        self.t = self.cont * SAMPLE_TIME
        agent.set_velocity(BACKWARD_SPEED, 0)
        self.cont += 1


class RotateState(State):
    def __init__(self):
        super().__init__("Rotate")
        # Todo: add initialization code
        self.cont = 0
        self.t = self.cont
        random.seed()
        self.rotate_angle = 2*math.pi*random.random() - math.pi

    def check_transition(self, agent, state_machine):
        # Todo: add logic to check and execute state transition
        if self.t > abs(self.rotate_angle/ANGULAR_SPEED):
            state_machine.change_state(MoveForwardState())
    
    def execute(self, agent):
        # Todo: add execution logic
        self.t = self.cont * SAMPLE_TIME
        agent.set_velocity(0, (2*(self.rotate_angle > 0) - 1)*ANGULAR_SPEED)
        self.cont += 1
