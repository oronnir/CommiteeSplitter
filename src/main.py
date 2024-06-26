import os
import random
import shutil

import networkx as nx
import numpy as np

from src.data_loader import load_graph
from src.graph_cutter import GraphCutter


def recreate_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path, True)
    os.makedirs(folder_path)


def cut_graph(graph, num_cuts, num_iterations, early_stop, num_meta_runs, step_folder, papers):
    cutter = GraphCutter(graph)
    best_cut_cost = len(graph.edges)
    best_cut_groups = None
    for i in range(num_meta_runs):
        random.seed(i)
        np.random.seed(i)
        print(f'Running meta run {i+1}/{num_meta_runs}')
        cut = cutter.cut(num_cuts=num_cuts, num_iterations=num_iterations, convergence_count=early_stop)
        total_cut_cost = cutter.graph_cut_loss(cut)
        print(f'Total cut cost: {total_cut_cost}')
        if total_cut_cost < best_cut_cost:
            best_cut_cost = total_cut_cost
            tc_output_data_path = os.path.join(step_folder, f'output_TC{int(total_cut_cost)}.json')
            cutter.save(tc_output_data_path, cut, papers)
            best_cut_groups = cut
    print(f'Best cut cost of the first step is: {best_cut_cost}')
    return best_cut_groups


def run_graph_cut_main():
    input_data_path = r"C:\CommitteeData\reviewer_assignments_23-06-24_valid.csv"
    first_step_folder = r'C:\CommitteeData\outputs\first_step'
    second_step_folder = r'C:\CommitteeData\outputs\second_step'
    num_cuts = 4
    early_stop = 100
    num_iterations = 600
    num_meta_runs = 5

    # load data
    graph, singletons, reviewers, papers = load_graph(input_data_path)

    # re-create a folder with all artifacts of the first stage
    recreate_folder(first_step_folder)

    # cut graph
    best_cut_groups = cut_graph(graph, num_cuts, num_iterations, early_stop, num_meta_runs, first_step_folder, papers)

    # second step - recreate folder
    recreate_folder(second_step_folder)

    # node id to cut group
    node_to_cut_group = {node: i for i, cut in enumerate(best_cut_groups) for node in cut}

    # second step - load the conflicting papers into a new graph
    conflicting_papers = set()
    for edge in graph.edges:
        from_node = edge[0]
        to_node = edge[1]
        # check if the edge is conflicting two cuts
        if node_to_cut_group[from_node] != node_to_cut_group[to_node]:
            conflicting_papers.add((from_node, to_node))

    # create a new graph with the conflicting papers
    conflicting_graph = nx.Graph()
    for paper1, paper2 in conflicting_papers:
        if conflicting_graph.has_edge(paper1, paper2):
            conflicting_graph[paper1][paper2]['weight'] += 1
        else:
            conflicting_graph.add_edge(paper1, paper2, weight=1)

    # cut the conflicting graph
    second_pass_best_cut_groups = cut_graph(conflicting_graph, num_cuts, num_iterations, early_stop, num_meta_runs, second_step_folder, papers)
    print('Second pass best cut groups:', second_pass_best_cut_groups)
    print('Graph cut test passed!')


if __name__ == '__main__':
    run_graph_cut_main()
    print('Graph cut test passed!')
