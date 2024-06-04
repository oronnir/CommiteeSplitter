from src.data_loader import load_graph
from src.graph_cutter import GraphCutter


def run_graph_cut_main():
    input_data_path = '../data/input.csv'
    input_data_path = r"C:\CommitteeData\assignmentsTPC.csv"
    output_data_path = 'graph_cut.json'
    num_cuts = 2
    early_stop = 1000
    num_iterations = 10000

    # load data
    graph, singletons, reviewers, papers = load_graph(input_data_path)

    # cut graph
    cutter = GraphCutter(graph)
    cut = cutter.cut(num_cuts=num_cuts, num_iterations=num_iterations, convergence_count=early_stop)

    # print cut cost
    total_cut_cost = cutter.graph_cut_loss(cut)
    print(f'Total cut cost: {total_cut_cost}')

    # save graph cut
    cutter.save(output_data_path, cut, papers)


if __name__ == '__main__':
    run_graph_cut_main()
    print('Graph cut test passed!')
