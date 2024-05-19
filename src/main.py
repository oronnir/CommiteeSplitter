import pickle
import time
from src.graph_cutter import GraphCutter


def run_grph_cut_main():
    input_data_path = 'graph.pkl'
    output_data_path = 'graph_cut.pkl'
    num_splits = 2
    num_iterations = 100

    print(f'Got input data: {input_data_path} and output data: {output_data_path}')

    # load data
    graph = pickle.load(open(input_data_path, 'rb'))

    print(f'Loaded graph with {len(graph)} nodes and {len(graph.edges)} edges.')
    print(f'Starting graph cut with {num_splits} splits and {num_iterations} iterations.')

    # cut graph
    cutter = GraphCutter(graph, num_splits=num_splits, num_iterations=num_iterations)
    start_time = time.time()
    cut = cutter.cut()
    end_time = time.time()
    print(f'Graph cut took {end_time - start_time} seconds.')

    # print cut cost
    total_cut_cost = cutter.graph_cut_loss(cut)
    print(f'Total cut cost: {total_cut_cost}')

    # save graph cut
    cutter.save(output_data_path)


if __name__ == '__main__':
    run_grph_cut_main()
    print('Graph cut test passed!')
