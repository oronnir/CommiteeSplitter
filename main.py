from src.data_loader import load_graph
from src.graph_cutter import GraphCutter


def run_grph_cut_main():
    input_data_path = 'data/input.csv'
    output_data_path = 'graph_cut.pkl'

    # load data
    graph, singletons = load_graph(input_data_path)

    # cut graph
    cutter = GraphCutter(graph, num_iterations=10000)
    cut = cutter.cut(2)

    # print cut cost
    total_cut_cost = cutter.graph_cut_loss(cut)
    print(f'Total cut cost: {total_cut_cost}')

    # save graph cut
    cutter.save(output_data_path, cut)


if __name__ == '__main__':
    run_grph_cut_main()
    print('Graph cut test passed!')
