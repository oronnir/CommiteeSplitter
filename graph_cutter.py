import pickle

import networkx as nx


class GraphCutter:
    def __init__(self, graph: nx.Graph, num_splits: int, num_iterations: int = 100):
        if num_splits < 2:
            raise ValueError("num_splits must be at least 2")
        self.graph = graph
        self.num_splits = num_splits
        self.num_iterations = num_iterations

    def graph_cut_loss(self, partition: list[nx.Graph]) -> float:
        """
        Calculate the cut cost of the partition.
        :param partition: the partition to calculate the cut cost of
        :return: the cut cost of the partition
        """
        running_total_cut_cost = 0
        for i in range(len(partition)):
            for j in range(i, len(partition)):
                running_total_cut_cost += nx.cut_size(self.graph, partition[i], partition[j], self.graph.edges)
        return running_total_cut_cost

    def cut(self) -> list[nx.Graph]:
        """
        Cut the graph into num_splits partitions while minimizing the cut cost.
        :return: self.num_splits partitions
        """
        # init by randomly assigning nodes to partitions in a balanced way
        partitions = [nx.Graph() for _ in range(self.num_splits)]
        for i, node in enumerate(self.graph.nodes):
            partitions[i % self.num_splits].add_node(node)

        # iteratively move nodes between partitions to minimize cut cost
        for _ in range(self.num_iterations):
            for node in self.graph.nodes:
                best_partition = None
                best_cut_cost = float('inf')
                for partition in partitions:
                    partition.add_node(node)
                    cut_cost = self.graph_cut_loss(partitions)
                    if cut_cost < best_cut_cost:
                        best_partition = partition
                        best_cut_cost = cut_cost
                    partition.remove_node(node)
                best_partition.add_node(node)
        return partitions

    def save(self, filepath: str):
        """
        Save the graph cut to a file.
        :param filepath: the path the file will be saved to
        :return: None
        """
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filepath: str) -> 'GraphCutter':
        """
        Load a graph cut from a file.
        :param filepath:
        :return:
        """
        with open(filepath, 'rb') as file:
            return pickle.load(file)

    def __str__(self):
        return f"GraphCutter(graph={self.graph}, num_splits={self.num_splits})"

    def __repr__(self):
        return str(self)
