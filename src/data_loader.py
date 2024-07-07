"""
Taken from https://github.com/adamfinkelstein/siggraph-pc-rooms/blob/main/gen-fake-data.py
Credit to Adam Finkelstein (Sep-Oct 2022)
"""
import networkx as nx
import pandas as pd


# Note: currently ignores withdrawn papers or those with <1 reviewer.
def read_assignments(paper_assignment_csv: str):
    score_threshold = -1
    reviewer_id_field_suffix = 'Email'  # 'Email' or 'Name'
    cols = ['Submission ID', 'Primary Name', 'Primary Email', 'Secondary Name', 'Secondary Email', 'Score']
    data = pd.read_csv(paper_assignment_csv, header=0, names=cols)

    reviewers = set()
    papers = {}
    singles = {}
    low_score_papers = {}

    for i, row in data.iterrows():
        pid = row['Submission ID']
        pri = row[f'Primary {reviewer_id_field_suffix}']
        sec = row[f'Secondary {reviewer_id_field_suffix}']
        score = row['Score']
        if not pid or pd.isna(pri) or pd.isna(sec) or pd.isna(score):
            singles[pid] = pri if not pd.isna(pri) else sec
            continue

        if score <= score_threshold:
            low_score_papers[pid] = (pri, sec)
            continue

        reviewers.add(pri)
        reviewers.add(sec)
        papers[pid] = (pri, sec)
    reviewers = sorted(reviewers, key=lambda r: int(r[1:]) if r[1:].isnumeric() else r)
    return reviewers, papers, singles, low_score_papers
    # with open(fname) as f:
    #     lines = f.readlines()
    # lines = lines[1:]  # skip header
    # for line in lines:
    #     parts = line.split(',')
    #     if len(parts) < 4:
    #         continue
    #     parts = [p.strip() for p in parts]  # remove surrounding whitespace
    #     pid, withdraw = parts[:2]
    #     if not pid or withdraw == 'True':
    #         continue
    #     revs = parts[2:]  # reviewers
    #     revs = [r for r in revs if len(r)]  # remove blank reviewers
    #     if len(revs) < 1:  # no reviewers
    #         print(f'skipping paper {pid} because no reviewers')
    #         continue
    #     if len(revs) < 2:  # just one reviewer
    #         singles[pid] = revs[0]
    #         continue
    #     pri = revs[0]
    #     sec = revs[1]
    #     reviewers.add(pri)
    #     reviewers.add(sec)
    #     papers[pid] = (pri, sec)
    # reviewers = sorted(reviewers, key=lambda r: int(r[1:]) if r[1:].isnumeric() else r)
    # return reviewers, papers, singles


def make_graph_from_paper_reviews(reviewers, papers):
    graph = nx.Graph()
    for r in reviewers:
        graph.add_node(r)
    for pid in papers:
        pri, sec = papers[pid]
        if graph.has_edge(pri, sec):
            graph[pri][sec]['weight'] += 1
            graph[pri][sec]['pids'].append(pid)
        else:
            graph.add_edge(pri, sec)
            graph[pri][sec]['weight'] = 1
            graph[pri][sec]['pids'] = [pid]
    graph_node_count = len(list(graph.nodes))
    graph_edge_count = len(list(graph.edges))
    print(f'Added {graph_node_count} nodes and {graph_edge_count} edges to graph.')
    return graph


def load_graph(file_path: str):
    print(f'Reading {file_path} ...')
    reviewers, papers, singles, low_score_papers = read_assignments(file_path)
    if len(singles) > 0:
        print(f'Warning: {len(singles)} papers have only one reviewer.')
    print('Input reviewers and papers:', len(reviewers), len(papers))
    graph = make_graph_from_paper_reviews(reviewers, papers)
    return graph, singles, reviewers, papers, low_score_papers
