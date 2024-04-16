from grid import Node, NodeGrid
from math import inf
import heapq


class PathPlanner(object):
    """
    Represents a path planner, which may use Dijkstra, Greedy Search or A* to plan a path.
    """
    def __init__(self, cost_map):
        """
        Creates a new path planner for a given cost map.

        :param cost_map: cost used in this path planner.
        :type cost_map: CostMap.
        """
        self.cost_map = cost_map
        self.node_grid = NodeGrid(cost_map)
        """
        Variables created to initialize algorithms
        """
        self.start_position = []
        self.goal_position = []
        self.pq = []
        self.start_cost = 0
        self.f = 0
        self.node = None
        self.successor_node = None
        self.goal_node = None

    @staticmethod
    def construct_path(goal_node):
        """
        Extracts the path after a planning was executed.

        :param goal_node: node of the grid where the goal was found.
        :type goal_node: Node.
        :return: the path as a sequence of (x, y) positions: [(x1,y1),(x2,y2),(x3,y3),...,(xn,yn)].
        :rtype: list of tuples.
        """
        node = goal_node
        # Since we are going from the goal node to the start node following the parents, we
        # are transversing the path in reverse
        reversed_path = []
        while node is not None:
            reversed_path.append(node.get_position())
            node = node.parent
        return reversed_path[::-1]  # This syntax creates the reverse list

    def dijkstra(self, start_position, goal_position):
        """
        Plans a path using the Dijkstra algorithm.

        :param start_position: position where the planning stars as a tuple (x, y).
        :type start_position: tuple.
        :param goal_position: goal position of the planning as a tuple (x, y).
        :type goal_position: tuple.
        :return: the path as a sequence of positions and the path cost.
        :rtype: list of tuples and float.
        """
        # Todo: implement the Dijkstra algorithm
        # The first return is the path as sequence of tuples (as returned by the method construct_path())
        # The second return is the cost of the path
        self.node_grid.reset()
        while len(self.pq) != 0:
            self.f, self.node = heapq.heappop(self.pq)
        self.start_position = start_position
        self.goal_position = goal_position
        self.goal_node = self.node_grid.get_node(self.goal_position[0], self.goal_position[1])
        self.node = self.node_grid.get_node(self.start_position[0], self.start_position[1])
        self.node.f = 0
        heapq.heappush(self.pq, (self.node.f, self.node))
        while len(self.pq) != 0:
            self.f, self.node = heapq.heappop(self.pq)
            if self.node.get_position() == self.goal_node.get_position():
                return self.construct_path(self.goal_node), self.node.f
            for successor in self.node_grid.get_successors(self.node.get_position()[0], self.node.get_position()[1]):
                self.successor_node = self.node_grid.get_node(successor[0], successor[1])
                if self.successor_node.f > self.node.f + self.cost_map.get_edge_cost(self.node.get_position(), self.successor_node.get_position()):
                    self.successor_node.f = self.node.f + self.cost_map.get_edge_cost(self.node.get_position(), self.successor_node.get_position())
                    self.successor_node.parent = self.node
                    heapq.heappush(self.pq, (self.successor_node.f, self.successor_node))

        return [], inf

    def greedy(self, start_position, goal_position):
        """
        Plans a path using greedy search.

        :param start_position: position where the planning stars as a tuple (x, y).
        :type start_position: tuple.
        :param goal_position: goal position of the planning as a tuple (x, y).
        :type goal_position: tuple.
        :return: the path as a sequence of positions and the path cost.
        :rtype: list of tuples and float.
        """
		# Todo: implement the Greedy Search algorithm
		# The first return is the path as sequence of tuples (as returned by the method construct_path())
		# The second return is the cost of the path
        self.node_grid.reset()
        while len(self.pq) != 0:
            self.f, self.node = heapq.heappop(self.pq)
        self.start_position = start_position
        self.goal_position = goal_position
        self.goal_node = self.node_grid.get_node(self.goal_position[0], self.goal_position[1])
        self.node = self.node_grid.get_node(self.start_position[0], self.start_position[1])
        self.node.f = 0
        self.node.g = 0
        heapq.heappush(self.pq, (self.node.f, self.node))
        while len(self.pq) != 0:
            self.f, self.node = heapq.heappop(self.pq)
            self.node.closed = True
            if self.node == self.goal_node:
                return self.construct_path(self.goal_node), self.node.g
            for successor in self.node_grid.get_successors(self.node.get_position()[0], self.node.get_position()[1]):
                self.successor_node = self.node_grid.get_node(successor[0], successor[1])
                if not self.successor_node.closed and self.successor_node.g >= inf:
                    self.successor_node.parent = self.node
                    self.successor_node.f = self.successor_node.distance_to(self.goal_position[0], self.goal_position[1])
                    self.successor_node.g = self.node.g + self.successor_node.distance_to(self.node.get_position()[0], self.node.get_position()[1])
                    heapq.heappush(self.pq, (self.successor_node.f, self.successor_node))

        return [], inf

    def a_star(self, start_position, goal_position):
        """
        Plans a path using A*.

        :param start_position: position where the planning stars as a tuple (x, y).
        :type start_position: tuple.
        :param goal_position: goal position of the planning as a tuple (x, y).
        :type goal_position: tuple.
        :return: the path as a sequence of positions and the path cost.
        :rtype: list of tuples and float.
        """
		# Todo: implement the A* algorithm
        # The first return is the path as sequence of tuples (as returned by the method construct_path())
        # The second return is the cost of the path
        self.node_grid.reset()
        while len(self.pq) != 0:
            self.f, self.node = heapq.heappop(self.pq)
        self.start_position = start_position
        self.goal_position = goal_position
        self.goal_node = self.node_grid.get_node(self.goal_position[0], self.goal_position[1])
        self.node = self.node_grid.get_node(self.start_position[0], self.start_position[1])
        self.node.g = 0
        self.node.f = self.node.distance_to(self.goal_position[0], self.goal_position[1])
        heapq.heappush(self.pq, (self.node.f, self.node))
        while len(self.pq) != 0:
            self.h, self.node = heapq.heappop(self.pq)
            self.node.closed = True
            if self.node.get_position() == self.goal_node.get_position():
                return self.construct_path(self.goal_node), self.node.g
            for successor in self.node_grid.get_successors(self.node.get_position()[0], self.node.get_position()[1]):
                self.successor_node = self.node_grid.get_node(successor[0], successor[1])
                if not self.successor_node.closed and self.successor_node.f > self.node.g + self.cost_map.get_edge_cost(self.node.get_position(), self.successor_node.get_position()) + self.successor_node.distance_to(self.goal_position[0], self.goal_position[1]):
                    self.successor_node.g = self.node.g + self.cost_map.get_edge_cost(self.node.get_position(), self.successor_node.get_position())
                    self.successor_node.f = self.successor_node.g + self.successor_node.distance_to(self.goal_position[0], self.goal_position[1])
                    self.successor_node.parent = self.node
                    heapq.heappush(self.pq, (self.successor_node.g, self.successor_node))
        return [], inf
