import pandas as pd
import numpy as np
import h5py
import networkx as nx
from scipy.sparse import coo_matrix 
import nnet_survival
from sklearn import preprocessing


def load_network(dataset_pth):
    ppi_network = pd.read_csv(dataset_pth, sep='\t')
    ppi_graph = nx.from_pandas_edgelist(df=ppi_network, source='partner1', target='partner2', edge_attr='confidence')
    ppi_network = nx.to_pandas_adjacency(G=ppi_graph)
    coo_ppi_network = coo_matrix(ppi_network)
    edge_index = [coo_ppi_network.row, coo_ppi_network.col]
    print('ppi_network have ',len(ppi_network.index), 'genes')
    return edge_index, ppi_network, ppi_graph


def load_exp(exp_pth):
    exp = pd.read_hdf(exp_pth,key='fc_sample_matrix') 
    exp = exp.loc[:, ~exp.columns.duplicated()]
    return exp

def load_clin(clin_pth):
    clin = pd.read_csv(clin_pth,sep='\t')
    return clin


def load_clear_edge_index(ppi_network):    
    ##get edge
    coo_ppi_network = coo_matrix(ppi_network)
    edge_index = [coo_ppi_network.row, coo_ppi_network.col]
    return edge_index

def load(args):
    ##load data from pth
    edge_index, ppi_network, _ = load_network(args.dataset_pth)
    exp = load_exp(args.exp_pth)
    clin = load_clin(args.clin_pth)
    return exp, clin, edge_index, ppi_network


def align_data(ppi_network,exp,clin):
    ge_nodes = exp[exp.index.isin(ppi_network.index)].shape[0]
    print ("* {} genes in network have gene expression".format(ge_nodes))
    ##aligin exp and ppi
    unique_id = list(set(exp.index).intersection(set(ppi_network.columns)))
    exp = exp.reindex(unique_id, fill_value=0)
    ppi_network = ppi_network.reindex(unique_id)
    ppi_network = ppi_network[unique_id]

    ##algin data
    unique_id = list(set(clin.index).intersection(set(exp.columns)))
    exp = exp[unique_id]
    clin = clin.loc[unique_id]
    print('exp shape',exp.shape,'ppi adj.shape',ppi_network.shape,'clin shape',clin.shape)    
    return exp,clin,ppi_network



def convert_vital_status(x):
    if(x == 'Dead'):
      return 1
    return 0

def make_data_for_model(args):

    ### load data
    exp, clin, edge_index, ppi_network, _ = load(args)

    exp,clin,ppi_network = align_data(ppi_network,exp,clin)
    edge_index = load_clear_edge_index(ppi_network)


    ### make label
    max_live_time = clin['days_to_death'].max()
    breaks_floor = max_live_time
    breaks=np.arange(0.,breaks_floor,breaks_floor/20)
    f = np.array(list(map(convert_vital_status, clin['vital_status'])))

    y_data = nnet_survival.make_surv_array(clin['times'],f,breaks)
    x_data = exp.values
    return x_data,y_data, (clin['times'],f,breaks),edge_index,ppi_network
