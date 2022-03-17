from util import create_graph
from tqdm import tqdm
import pickle
from train import run_mapper, get_args


def create_graphs():
    lnode = [10, 20, 30, 40, 50]
    graphs = {}
    for nnodes in tqdm(lnode):
        graphs[nnodes] = []
        for samples in range(100):
            graphdef = create_graph(None, numnodes = nnodes)
            graphs[nnodes].append(graphdef)

    with open("graphs.pkl","wb") as file:
       pickle.dump(graphs, file)


if __name__ == "__main__":
    args = get_args()  # Holds all the input arguments
    args.quiet = True
    args.epochs = 50000
    args.device_topology = [16, 6]

    with open("graphs.pkl", "rb") as file:
        dataset = pickle.load(file)

    results = {}
    for num_node in tqdm(dataset):
        results[num_node] = {'time':[], 'reward':[]}
        for i in range(0, 5):
            time, reward = run_mapper(args, dataset[num_node][i])
            results[num_node]['time'].append(time)
            results[num_node]['reward'].append(reward)

        save = "results"+num_node+".pkl"
        with open(save,"wb") as file:
           pickle.dump(results, file)
