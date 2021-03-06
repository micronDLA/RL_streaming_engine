import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import scipy
from scipy import signal
import pickle
from plotly.subplots import make_subplots
import torch
import torch.nn.functional as F
import plotly.express as px

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

def plot_attn():
    fname = '../att_/attn_150000_'
    all_data = []
    key ='attn'
    for i in range(0, 17):
        with open(fname+str(i)+'.pkl', "rb") as file:
            data = pickle.load(file)
            all_data.append(data[key])

    fig = make_subplots(2,9)
    for i in range(0, 8):
        d = torch.sum(all_data[i], dim=0, keepdim=True)
        d = F.softmax(d, dim=1)
        fig.add_trace(go.Heatmap(z = d, zmin=0, zmax=1, colorscale = 'Viridis', y=[str(i)]), 1, i+1)
        fig.layout.height = 500
    for i in range(8, 16):
        d = torch.sum(all_data[i], dim=0, keepdim=True)
        d = F.softmax(d, dim=1)
        fig.add_trace(go.Heatmap(z = d, zmin=0, zmax=1, colorscale = 'Viridis', y=[str(i)]), 2, i+1-8)
        fig.layout.height = 500
    d = torch.sum(all_data[16], dim=0, keepdim=True)
    d = F.softmax(d, dim=1)
    fig.add_trace(go.Heatmap(z = d, zmin=0, zmax=1, colorscale = 'Viridis', y=['16']), 2, 9)
    fig.layout.height = 500
    fig.update_layout(coloraxis = {'colorscale':'viridis'})
    fig.show()
    fig.write_image("att.pdf")


def plot_topologicalorder():
    df = pd.read_csv('with_without_topologicalorder_nodeplace_ff_gnn_attention/no_ordered_data.csv')
    df2 = pd.read_csv('with_without_topologicalorder_nodeplace_ff_gnn_attention/ordered_data.csv')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df2['Step'], y=smooth(df2['Value'], 0.9),
        name="Ordered"
    ))
    fig.add_trace(go.Scatter(
        x=df['Step'], y=smooth(df['Value'], 0.1),
        name="Not Ordered"
    ))

    fig.update_layout(
        xaxis_title="Epochs",
        yaxis_title="Rewards",
        xaxis_range=[2000, 15000],
        showlegend=True,
        font=dict(
            family="Courier New, monospace",
            size=20,
        )
    )
    fig.show()
    fig.write_image("topologicalorder.pdf")

def plot_ppo():
    df = pd.read_csv('ifft_reward/ppo_data.csv')
    df2 = pd.read_csv('ifft_reward/ppo_gnn_transform_data.csv')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df2['Step'], y=smooth(df2['Value'], 0.99),
        name="GGA+MLP"
    ))
    fig.add_trace(go.Scatter(
        x=df['Step'], y=smooth(df['Value'], 0.99),
        name="MLP"
    ))
    fig.update_layout(
        xaxis_title="Epochs",
        yaxis_title="Rewards",
        xaxis_range=[10000, 150000],
        showlegend=True,
        font=dict(
            family="Courier New, monospace",
            size=20,
        )
    )
    fig.show()
    fig.write_image("ppo.pdf")

def plot_masking():
    df = pd.read_csv('with_nomask/ff_gnn_ifft_loop_with_mask.csv')
    df2 = pd.read_csv('with_nomask/ff_gnn_ifft_loop_without_mask2.csv')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Step'], y=smooth(df['Value'], 0.99),
        name="With Mask"
    ))
    fig.add_trace(go.Scatter(
        x=df2['Step'], y=smooth(df2['Value'], 0.99),
        name="Without Mask"
    ))
    fig.update_layout(
        xaxis_title="Epochs",
        yaxis_title="Rewards",
        xaxis_range=[2000, 100000],
        showlegend=True,
        font=dict(
            family="Courier New, monospace",
            size=20,
        )
    )
    fig.show()
    fig.write_image("masking.pdf")



def plot_scaling_nodes():
    df = pd.read_csv('nodes_data.csv')
    print(df)
    fig = px.bar(df, x='nodes', y=['MLP', 'GGA+MLP'], barmode='group')
    fig.update_xaxes(type='category')
    fig.update_layout(
        title = "      16 tiles                                                        64 tiles",
        xaxis_title="Number of Nodes",
        yaxis_title="Best schedule cycle count",
        showlegend=True,
    )
    fig.show()
    fig.write_image("scaling_nodes.pdf")

def plot_pretrain():
    df = pd.read_csv('with_pretrain/run_after_pretrain_ifft_loop.csv')
    df2 = pd.read_csv('with_pretrain/run_ff_gnn_attention_ifft_nopretrain.csv')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Step'], y=smooth(df['Value'], 0.99),
        name="fine-tuned"
    ))
    fig.add_trace(go.Scatter(
        x=df2['Step'], y=smooth(df2['Value'], 0.99),
        name="scratch"
    ))
    fig.update_layout(
        xaxis_title="Epochs",
        yaxis_title="Rewards",
        xaxis_range=[1000, 70000],
        showlegend=True,
        font=dict(
            family="Courier New, monospace",
            size=20,
        )
    )
    fig.show()
    fig.write_image("pretrain.pdf")

def plot_sa():
    df = pd.read_csv('experiments/compare_sa/ppo_gnn_transform_data.csv')
    df2 = pd.read_csv('experiments/compare_sa/simulated_anneal_ifft.csv')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Step'], y=smooth(df['Value'], 0.99),
        name="GGA+MLP"
    ))
    fig.add_trace(go.Scatter(
        x=df2['Step'], y=smooth(df2['Value'], 0.99),
        name="SA"
    ))
    fig.update_layout(
        xaxis_title="Epochs",
        yaxis_title="Rewards",
        xaxis_range=[2000, 100000],
        showlegend=True,
        font=dict(
            family="Courier New, monospace",
            size=20,
        )
    )
    fig.show()
    fig.write_image("sa.pdf")


if __name__ == "__main__":
    pass