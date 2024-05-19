import pickle
from src.graph_cutter import GraphCutter


def run_grph_cut_main():
    input_data_path = 'graph.pkl'
    output_data_path = 'graph_cut.pkl'

    # load data
    graph = pickle.load(open(input_data_path, 'rb'))

    # cut graph
    cutter = GraphCutter(graph, num_splits=2, num_iterations=100)
    cut = cutter.cut()

    # print cut cost
    total_cut_cost = cutter.graph_cut_loss(cut)
    print(f'Total cut cost: {total_cut_cost}')

    # save graph cut
    cutter.save(output_data_path)

if __name__ == '__main__':
    run_grph_cut_main()
    print('Graph cut test passed!')
