import json
import numpy as np
import networkx as nx


class GraphCutter:
    def __init__(self, graph: nx.Graph, num_iterations: int = 100):
        self.graph = graph
        self.num_iterations = num_iterations

    def graph_cut_loss(self, partitions: list[nx.Graph]) -> float:
        """
        Sums the weight of edges in the k-cut of a graph using adjacency matrices.

        Parameters:
        - G (networkx.Graph): The input graph.
        - partitions (list of lists/sets): The k partitions of the graph. Each element is a list or set of nodes forming a partition.

        Returns:
        - float: The sum of the weights of the edges in the k-cut.
        """
        # Create the adjacency matrix of the graph
        weights = nx.to_numpy_array(self.graph)

        # Get the list of all nodes
        all_nodes = list(self.graph.nodes())

        # Iterate over each pair of partitions
        cut_weight = 0
        for i in range(len(partitions)):
            for j in range(i + 1, len(partitions)):
                # Get the indices of the nodes in each partition
                indices_i = [all_nodes.index(node) for node in partitions[i]]
                indices_j = [all_nodes.index(node) for node in partitions[j]]

                # Extract the sub-matrix that corresponds to the edges between the partitions
                cut_matrix = weights[indices_i, :][:, indices_j]

                # Sum the weights of the cut edges between these partitions
                cut_weight += np.sum(cut_matrix)

        return cut_weight

    def _init_cost_node_to_partition(self, num_cuts: int, partitions: list[nx.Graph], node_to_id_map: dict):
        cost_node_to_partition = np.zeros((len(self.graph.nodes), num_cuts))
        node_to_partition_map = {}
        for p, partition in enumerate(partitions):
            for node in partition.nodes:
                node_to_partition_map[node] = p

        for edge in self.graph.edges:
            cost_node_to_partition[node_to_id_map[edge[0]], node_to_partition_map[edge[1]]] += self.graph[edge[0]][edge[1]]['weight']

        return cost_node_to_partition, node_to_partition_map

    def _init_partition(self, num_cuts: int):
        if num_cuts < 2:
            raise ValueError("num_splits must be at least 2")

        nodes = sorted(self.graph.nodes, key=lambda n: int(n[1:]))
        node_to_id_map = {node: i for i, node in enumerate(nodes)}
        id_to_node_map = {i: node for i, node in enumerate(nodes)}

        # init by randomly assigning nodes to partitions in a balanced way
        partitions = [nx.Graph() for _ in range(num_cuts)]
        for i, node in enumerate(self.graph.nodes):
            partitions[i % num_cuts].add_node(node)

        return partitions, node_to_id_map, id_to_node_map

    def cut(self, num_cuts, convergence_count=50) -> list[nx.Graph]:
        """
        Cut the graph into num_splits partitions while minimizing the cut cost.
        :return: self.num_splits partitions
        """
        convergence_counter = convergence_count
        partitions, node_to_id_map, id_to_node_map = self._init_partition(num_cuts)

        # iteratively move nodes between partitions to minimize cut cost
        it = 0
        for it in range(self.num_iterations):
            cost_node_to_partition, node_to_partition_map = self._init_cost_node_to_partition(num_cuts, partitions, node_to_id_map)

            # pick two random partitions without replacement
            from_partition, to_partition = np.random.choice(num_cuts, 2, replace=False)
            swap_weights = cost_node_to_partition[:, from_partition] - cost_node_to_partition[:, to_partition]

            # mask from partition nodes
            mask_from = np.zeros(len(self.graph.nodes), dtype=bool)
            mask_from[[node_to_id_map[node] for node in partitions[from_partition].nodes]] = True
            masked_swap_from = np.ma.masked_array(swap_weights, mask_from, dtype=int)

            # mask to partition nodes
            mask_to = np.zeros(len(self.graph.nodes), dtype=bool)
            mask_to[[node_to_id_map[node] for node in partitions[to_partition].nodes]] = True
            masked_swap_to = np.ma.masked_array(swap_weights, mask_to, dtype=int)

            # calculate the current cost on crossing the partitions
            current_cost = self.graph_cut_loss(partitions)
            print(f'Iteration {it + 1}/{self.num_iterations}; Total Cut Cost: {current_cost}')

            # find the node with the highest cost to partition and swap it with the node with the highest cost from
            # partition so the cost is minimized and the partitions are kept balanced
            max_cost_node = np.ma.argmax(masked_swap_from)
            min_cost_node = np.ma.argmin(masked_swap_to)

            if swap_weights[max_cost_node] - swap_weights[min_cost_node] > 0:
                partitions[to_partition].remove_node(id_to_node_map[max_cost_node])
                partitions[from_partition].add_node(id_to_node_map[max_cost_node])
                partitions[from_partition].remove_node(id_to_node_map[min_cost_node])
                partitions[to_partition].add_node(id_to_node_map[min_cost_node])
                convergence_counter = convergence_count
            else:
                convergence_counter -= 1
                if convergence_counter == 0:
                    break
        print(f'Converged after {it+1} iterations. Into {num_cuts} partitions.')
        return partitions

    @staticmethod
    def save(filepath: str, cuts: list[nx.Graph]):
        """
        Save the graph cut to a file.
        :param filepath: the path the file will be saved to
        :param cuts: the cuts to save
        :return: None
        """
        reviewer_to_room_map = {node: i for i, cut in enumerate(cuts) for node in cut.nodes}
        with open(filepath, 'wb') as file:
            json.dump(reviewer_to_room_map, file)

    def __str__(self):
        return f"GraphCutter(graph={self.graph})"

    def __repr__(self):
        return str(self)
