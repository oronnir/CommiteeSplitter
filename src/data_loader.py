"""
Taken from https://github.com/adamfinkelstein/siggraph-pc-rooms/blob/main/gen-fake-data.py
Credit to Adam Finkelstein (Sep-Oct 2022)
"""
import networkx as nx


# Note: currently ignores withdrawn papers or those with <1 reviewer.
def read_assignments(fname):
    reviewers = set()
    papers = {}
    singles = {}
    with open(fname) as f:
        lines = f.readlines()
    lines = lines[1:]  # skip header
    for line in lines:
        parts = line.split(',')
        if len(parts) < 4:
            continue
        parts = [p.strip() for p in parts]  # remove surrounding whitespace
        pid, withdraw = parts[:2]
        if not pid or withdraw == 'True':
            continue
        revs = parts[2:]  # reviewers
        revs = [r for r in revs if len(r)]  # remove blank reviewers
        if len(revs) < 1:  # no reviewers
            print(f'skipping paper {pid} because no reviewers')
            continue
        if len(revs) < 2:  # just one reviewer
            singles[pid] = revs[0]
            continue
        pri = revs[0]
        sec = revs[1]
        reviewers.add(pri)
        reviewers.add(sec)
        papers[pid] = (pri, sec)
    reviewers = sorted(reviewers, key=lambda r: int(r[1:]) if r[1:].isnumeric() else r)
    return reviewers, papers, singles


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
    reviewers, papers, singles = read_assignments(file_path)
    if len(singles) > 0:
        print(f'Warning: {len(singles)} papers have only one reviewer.')
    print('Input reviewers and papers:', len(reviewers), len(papers))
    graph = make_graph_from_paper_reviews(reviewers, papers)
    return graph, singles, reviewers, papers
