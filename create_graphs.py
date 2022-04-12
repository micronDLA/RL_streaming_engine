from util import create_graph
from tqdm import tqdm
import pickle
from train import run_mapper, get_args
from coolname import generate_slug
from torch.utils.tensorboard import SummaryWriter

# create rand graphs and save into file for benchmark
def create_graphs():
    lnode = [10, 15, 20, 30, 40]
    graphs = {}
    for nnodes in tqdm(lnode):
        graphs[nnodes] = []
        for samples in range(100):
            graphdef = create_graph(None, numnodes = nnodes)
            graphs[nnodes].append(graphdef)

    with open("graphs.pkl","wb") as file:
       pickle.dump(graphs, file)

def ex_topoorder():
    args = get_args()  # Holds all the input arguments
    args.epochs = 500000
    args.device_topology = [16, 6]
    args.nnmode = 'ff_gnn_attention'

    with open("graphs_new2.pkl", "rb") as file:
        dataset = pickle.load(file)

    results = {}
    num_node = 15
    results[num_node] = {'time':[], 'reward':[]}
    time, reward = run_mapper(args, dataset[num_node][0])
    results[num_node]['time'].append(time)
    results[num_node]['reward'].append(reward)

    save = "results"+str(num_node)+".pkl"
    with open(save,"wb") as file:
       pickle.dump(results, file)


def ex_scale_nodes():
    args = get_args()  # Holds all the input arguments
    args.quiet = True
    args.epochs = 50000
    args.device_topology = [64, 6]
    args.nnmode = 'simple_ff'

    with open("graphs_new2.pkl", "rb") as file:
        dataset = pickle.load(file)

    results = {}
    num_nodes = [10, 15, 20, 30, 40]
    for num_node in tqdm(num_nodes):
        results[num_node] = {'time':[], 'reward':[]}
        for i in range(0, 5):
            time, reward = run_mapper(args, dataset[num_node][i])
            results[num_node]['time'].append(time)
            results[num_node]['reward'].append(reward)

        save = "results_"+str(num_node)+".pkl"
        with open(save,"wb") as file:
           pickle.dump(results, file)

def ex_curriculum_rl():
    writer = SummaryWriter(comment=f'_{generate_slug(2)}')
    print(f'[INFO] Saving log data to {writer.log_dir}')
    writer.add_text('experiment config', 'ex_curriculum_rl')
    writer.flush()

    with open("graphs_new2.pkl", "rb") as file:
        dataset = pickle.load(file)

    args = get_args()  # Holds all the input arguments
    args.epochs = 20000
    args.device_topology = [16, 6]
    args.nnmode = 'ff_gnn_attention'
    num_node = 10
    run_mapper(args, dataset[num_node], writer)

    args = get_args()  # Holds all the input arguments
    args.epochs = 20000
    args.device_topology = [16, 6]
    args.nnmode = 'ff_gnn_attention'
    args.model = 'model_epoch.pth'
    num_node = 15
    run_mapper(args, dataset[num_node], writer)

    args = get_args()  # Holds all the input arguments
    args.epochs = 50000
    args.device_topology = [16, 6]
    args.nnmode = 'ff_gnn_attention'
    args.model = 'model_epoch.pth'
    num_node = 20
    run_mapper(args, dataset[num_node], writer)

    args = get_args()  # Holds all the input arguments
    args.epochs = 100000
    args.device_topology = [16, 6]
    args.nnmode = 'ff_gnn_attention'
    args.model = 'model_epoch.pth'
    num_node = 30
    results = {}
    results[num_node] = {'time': [], 'reward': []}
    time, reward = run_mapper(args, dataset[num_node], writer)
    results[num_node]['time'].append(time)
    results[num_node]['reward'].append(reward)


if __name__ == "__main__":
    ex_topoorder()