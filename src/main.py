import os
import random
import shutil
from collections import Counter
from collections import defaultdict
import pandas as pd
import networkx as nx
import numpy as np
from tqdm import tqdm
from src.data_loader import load_graph
from src.graph_cutter import GraphCutter
import traceback


def recreate_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path, True)
    os.makedirs(folder_path)


def cut_graph(graph, num_cuts, num_iterations, early_stop, num_meta_runs, step_folder, papers, reviewer_to_areas):
    if reviewer_to_areas is None:
        init_reviewer_to_group = None
    else:
        # use the areas to cut the graph as an intial guess
        area_counter = Counter(reviewer_to_areas.values())
        popular_areas = {area[0]: i for i, area in enumerate(sorted(list(area_counter.items()), key=lambda t: t[1], reverse=True)[:num_cuts-1])}
        area_codes = defaultdict(lambda: num_cuts-1, popular_areas)
        init_reviewer_to_group = {reviewer: area_codes[reviewer_to_areas[reviewer]] for reviewer in graph.nodes}

    cutter = GraphCutter(graph)
    # sum weights of all edges
    best_cut_cost = sum(graph.edges[edge]['weight'] for edge in graph.edges)
    best_cut_groups = None
    for i in tqdm(range(num_meta_runs)):
        random.seed(i)
        np.random.seed(i)
        print(f'Running meta run {i+1}/{num_meta_runs}')
        cut = cutter.cut(num_cuts=num_cuts, num_iterations=num_iterations, convergence_count=early_stop, init_guess=init_reviewer_to_group)
        total_cut_cost = cutter.graph_cut_loss(cut)
        print(f'Total cut cost: {total_cut_cost}')
        if total_cut_cost < best_cut_cost:
            best_cut_cost = total_cut_cost
            tc_output_data_path = os.path.join(step_folder, f'output_TC{int(total_cut_cost)}.json')
            try:
                cutter.save(tc_output_data_path, cut, papers)
            except Exception as e:
                print(f'Failed to save the cut data to {tc_output_data_path}.')
                # print stack trace
                traceback.print_exc()
                raise e
            best_cut_groups = cut
    print(f'Best cut cost of this step is: {best_cut_cost}')
    return best_cut_groups, best_cut_cost


def find_graph_conflicts(graph, papers, best_cut_groups):

    # node id to cut group
    node_to_cut_group = {node: i for i, cut in enumerate(best_cut_groups) for node in cut}

    # second step - load the conflicting papers into a new graph
    print('Starting building the conflicting graph')
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

    # update the conflicting graph papers
    conflicting_papers_to_reviewers = {p: (r1, r2) for p, (r1, r2) in papers.items() if (r1, r2) in conflicting_papers or (r2, r1) in conflicting_papers}

    # cut groups without conflicting papers and reviewers
    nonconflicting_reviewers_to_groups = [cut for cut in best_cut_groups if not any(node in conflicting_papers for node in cut)]
    nonconflicting_papers_to_groups = {p: (r1, r2) for p, (r1, r2) in papers.items() if (r1, r2) not in conflicting_papers and (r2, r1) not in conflicting_papers}

    return conflicting_graph, conflicting_papers_to_reviewers, nonconflicting_reviewers_to_groups, nonconflicting_papers_to_groups


def serialize_two_step_results(first_step_groups, second_step_groups, nonconflicting_papers_to_groups_step1,
                               nonconflicting_papers_to_groups_step2, conflicts_step2, reviewer_name, output_path):
    """
    Serialize the two-step results to a CSV file where each reviewer is assigned to a paper and a room.
    :param first_step_groups:
    :param second_step_groups:
    :param nonconflicting_papers_to_groups_step1:
    :param nonconflicting_papers_to_groups_step2:
    :param conflicts_step2: the conflicts after step 1
    :param reviewer_name: mapping the reviewers to their names
    :param output_path: output repo
    :return:
    """
    # validate the groups are disjoint
    for i, group1 in enumerate(first_step_groups):
        for group2 in first_step_groups[i+1:]:
            assert len(set(group1).intersection(group2)) == 0

    for i, group1 in enumerate(second_step_groups):
        for group2 in second_step_groups[i+1:]:
            assert len(set(group1).intersection(group2)) == 0

    csv_col_names = 'Paper Id,Primary Reviewer Email,Primary Reviewer Name,Secondary Reviewer Email,Secondary Reviewer Name,Discussion Room\n'

    # write a CSV file with the first step non-conflicting papers discussion rooms and the reviewers assigned to them
    first_step_nonconflict_path = os.path.join(output_path, 'first_step.csv')
    with open(first_step_nonconflict_path, 'w') as file:
        file.write(csv_col_names)
        step1_reviewer_to_group = {reviewer: i for i, group in enumerate(first_step_groups) for reviewer in group}
        for paper, (r1, r2) in nonconflicting_papers_to_groups_step1.items():
            # validate both reviewers are in the same room
            r1_room = step1_reviewer_to_group[r1]
            r2_room = step1_reviewer_to_group[r2]
            assert r1_room == r2_room
            file.write(f'{paper},{r1},{reviewer_name[r1]},{r2},{reviewer_name[r2]},{r1_room}\n')

    # write a CSV file with the second step non-conflicting papers discussion rooms and the reviewers assigned to them
    second_step_nonconflict_path = os.path.join(output_path, 'second_step.csv')
    with open(second_step_nonconflict_path, 'w') as file:
        file.write(csv_col_names)
        step2_reviewer_to_group = {reviewer: i for i, group in enumerate(second_step_groups) for reviewer in group}
        for paper, (r1, r2) in nonconflicting_papers_to_groups_step2.items():
            # validate both reviewers are in the same room
            r1_room = step2_reviewer_to_group[r1]
            r2_room = step2_reviewer_to_group[r2]
            assert r1_room == r2_room
            file.write(f'{paper},{r1},{reviewer_name[r1]},{r2},{reviewer_name[r2]},{r1_room}\n')

    # write a CSV file with the conflicting papers discussion after the second step
    conflicts_path = os.path.join(output_path, 'conflicts.csv')
    with open(conflicts_path, 'w') as file:
        file.write(csv_col_names)
        for paper, (r1, r2) in conflicts_step2.items():
            # validate both reviewers are not in the same room
            r1_room = step2_reviewer_to_group.get(r1, None)
            r2_room = step2_reviewer_to_group.get(r2, None)
            if r1_room is not None and r2_room is not None:
                assert r1_room != r2_room
            file.write(f'{paper},{r1},{reviewer_name[r1]},{r2},{reviewer_name[r2]},{-1}\n')

    # write a CSV file with reviewer to room assignment stage 1
    reviewer_to_room_path = os.path.join(output_path, 'reviewer_to_room_stage1.csv')
    with open(reviewer_to_room_path, 'w') as file:
        file.write('Reviewer,Name,Room\n')
        for reviewer, room in step1_reviewer_to_group.items():
            file.write(f'{reviewer},{reviewer_name[reviewer]},{room}\n')

    # write a CSV file with reviewer to room assignment stage 2
    reviewer_to_room_path = os.path.join(output_path, 'reviewer_to_room_stage2.csv')
    with open(reviewer_to_room_path, 'w') as file:
        file.write('Reviewer,Name,Room\n')
        for reviewer, room in step2_reviewer_to_group.items():
            file.write(f'{reviewer},{reviewer_name[reviewer]},{room}\n')


def update_with_low_score_papers(low_score_papers, nonconflicting_groups, nonconflicting_papers_to_groups):
    valid_low_score_papers_to_exclude = set()
    for paper, (r1, r2) in low_score_papers.items():
        if paper in nonconflicting_papers_to_groups:
            continue
        group_id = -1
        for g_idx, group in enumerate(nonconflicting_groups):
            if r1 in group and r2 in group:
                group_id = g_idx
                break
        if group_id != -1:
            nonconflicting_papers_to_groups[paper] = (r1, r2)
            nonconflicting_groups[group_id].add(r1)
            nonconflicting_groups[group_id].add(r2)
            valid_low_score_papers_to_exclude.add(paper)

    num_valid_low_score_papers = len(valid_low_score_papers_to_exclude)
    for paper in valid_low_score_papers_to_exclude:
        del low_score_papers[paper]
    print(f'Excluded {num_valid_low_score_papers} low score papers from the conflicting papers')
    return low_score_papers, nonconflicting_groups, nonconflicting_papers_to_groups


def update_conflicts_low_score_papers(second_conflicting_papers_to_reviewers, reviewers_to_names, low_score_papers):
    print(f'Number of low score papers on the cut: {len(low_score_papers)}')
    print(f'Number of high score papers on the cut: {len(second_conflicting_papers_to_reviewers)}')
    for paper, (r1, r2) in low_score_papers.items():
        if paper in second_conflicting_papers_to_reviewers:
            continue
        second_conflicting_papers_to_reviewers[paper] = (r1, r2)
        if r1 not in reviewers_to_names:
            reviewers_to_names[r1] = 'Low Score Reviewer'
        if r2 not in reviewers_to_names:
            reviewers_to_names[r2] = 'Low Score Reviewer'
    return second_conflicting_papers_to_reviewers, reviewers_to_names


def run_graph_cut_main():
    reviewer_metadata_xlsx_path = r"C:\CommitteeData\ReviewerAreas.xlsx"

    # input_data_path = r"C:\CommitteeData\reviewer_assignments_23-06-24_valid.csv"
    input_data_path = r"C:\CommitteeData\reviewer_assignments_scores_07-07-24.csv"
    first_step_folder = r'C:\CommitteeData\outputs\first_step'
    second_step_folder = r'C:\CommitteeData\outputs\second_step'
    final_output_path = r'C:\CommitteeData\outputs\final_output'
    num_cuts = 4
    early_stop = 200
    num_iterations = 600
    num_meta_runs = 500

    # read reviewer metadata
    reviewer_metadata = pd.read_excel(reviewer_metadata_xlsx_path)
    reviewers_to_names = {row['Reviewer']: row['First Name'] + " " + row['Last Name'] for _, row in reviewer_metadata.iterrows()}
    reviewer_to_areas = {row['Reviewer']: row['Area'] for _, row in reviewer_metadata.iterrows()}

    # load data
    graph, singletons, reviewers, papers, low_score_papers = load_graph(input_data_path)

    # re-create a folder with all artifacts of the first stage
    recreate_folder(first_step_folder)

    # cut graph
    print('Starting graph cut step 1')
    best_cut_groups, first_step_cost = cut_graph(graph, num_cuts, num_iterations, early_stop, num_meta_runs, first_step_folder, papers, reviewer_to_areas)

    # find conflicting papers
    conflicting_graph, conflicting_papers_to_reviewers, nonconflicting_groups, nonconflicting_papers_to_groups = find_graph_conflicts(graph, papers, best_cut_groups)

    # second step - recreate folder
    recreate_folder(second_step_folder)

    # cut the conflicting graph
    print('Starting graph cut step 2')
    second_pass_best_cut_groups, second_step_cost = cut_graph(conflicting_graph, num_cuts, num_iterations, early_stop,
                                                              num_meta_runs, second_step_folder, conflicting_papers_to_reviewers, None)
    print('Second pass best cut groups:', second_pass_best_cut_groups)
    print('Second pass best cut cost:', second_step_cost)

    # find conflicting papers
    second_conflicting_graph, second_conflicting_papers_to_reviewers, second_nonconflicting_groups, second_nonconflicting_papers_to_groups = find_graph_conflicts(conflicting_graph, conflicting_papers_to_reviewers, second_pass_best_cut_groups)

    # update the final output objects with the low score papers
    low_score_papers, nonconflicting_groups, nonconflicting_papers_to_groups = update_with_low_score_papers(low_score_papers, nonconflicting_groups, nonconflicting_papers_to_groups)
    low_score_papers, second_nonconflicting_groups, second_nonconflicting_papers_to_groups = update_with_low_score_papers(low_score_papers, second_nonconflicting_groups, second_nonconflicting_papers_to_groups)
    second_conflicting_papers_to_reviewers, reviewers_to_names = update_conflicts_low_score_papers(second_conflicting_papers_to_reviewers, reviewers_to_names, low_score_papers)

    # serialize the results
    recreate_folder(final_output_path)
    serialize_two_step_results(nonconflicting_groups, second_nonconflicting_groups, nonconflicting_papers_to_groups,
                               second_nonconflicting_papers_to_groups, second_conflicting_papers_to_reviewers,
                               reviewers_to_names, final_output_path)
    print('Graph cut test passed!')


if __name__ == '__main__':
    run_graph_cut_main()
    print('Graph cut test passed!')

