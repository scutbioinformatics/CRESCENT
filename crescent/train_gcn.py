# %%
from itertools import *
import prepare_data
import nnet_survival
import models
import numpy as np
import torch
from torch import optim
import torch.utils.data as Data
from lifelines.utils import concordance_index
from parser import Parser
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
from utils import normalize,sparse_mx_to_torch_sparse_tensor


import warnings
warnings.filterwarnings('ignore')

def get_Data(device,args):
    x_train,y_train, (time,event,breaks),edge_index,ppi_network = prepare_data.make_data_for_model(args)
    ## numpy to tensor 
    x = torch.tensor(x_train.T).float()
    y = torch.tensor(y_train).float()

    time = torch.tensor(time)
    event = torch.tensor(event)

    data = np.ones(edge_index[1].shape[0])
    adj = sp.coo_matrix((data, (edge_index[0], edge_index[1])), shape=(ppi_network.shape[0], ppi_network.shape[1]), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)


    ###load data to GPU
    x = x.to(device)
    y = y.to(device)


    ###
    adj = adj.to_dense()

    count = torch.sign(adj).sum()
    print('-------Here!',count - adj.shape[0],'edges on adj',adj.shape,'------')

    adj = adj.to(device)
    time = time.to(device)
    event = event.to(device)

    ## split data set
    gene_dataset = Data.TensorDataset(x,y,time,event)

    train_size = int(len(gene_dataset)*0.8)
    test_size = len(gene_dataset) - train_size


    train_dataset, test_dataset = torch.utils.data.random_split(gene_dataset, [train_size , test_size])
    print('len of train_size',train_size,'len of test_size',test_size)

    train_loader = Data.DataLoader(
        dataset=train_dataset,      # torch TensorDataset format
        batch_size=args.batch_size,      # mini batch size
        shuffle=True,                        
    )

    test_loader = Data.DataLoader(
        dataset=test_dataset,      # torch TensorDataset format
        batch_size=args.batch_size,     
        # shuffle=True,              
    )


    node_nums = x_train.shape[0]
    n_intervals = int(y_train.shape[1]/2)

    return train_loader, test_loader, adj, node_nums, n_intervals, breaks, train_dataset,test_dataset

def train_and_eval(
    model,
    adj,
    loss_func,
    rank_loss,
    optimizer,
    scheduler,
    breaks,
    train_loader,
    test_loader,
    args,
    train_dataset,
    test_dataset
    ):

    train_size = len(train_dataset.indices)
    test_size = len(test_dataset.indices)

    LOSS = []
    num_epoch = args.epochs
    TEST_max_C_TD = 0



    for epoch in range(num_epoch): 
        model.train()
        loop = tqdm(enumerate(train_loader), total =len(train_loader))
        y_preds = 0
        y_trues = 0
        times = 0
        events = 0
        
        epoch_loss = []
        for step, (batch_x,y_true,time,event) in loop:

            y_pred = model(adj,batch_x)            
            loss1 = loss_func(y_true, y_pred).mean()  
            loss2 = rank_loss(y_true, y_pred,time,event)
            loss = loss1 + args.b*loss2

            optimizer.zero_grad()   
            loss.mean().backward()      
            optimizer.step()

            y_pred = y_pred.cpu().detach().numpy()
            y_true = y_true.cpu().detach().numpy()
            time = time.cpu().detach().numpy()
            event = event.cpu().detach().numpy()
            if(step == 0):
                y_preds = y_pred
                y_trues = y_true
                times = time
                events = event
            else:
                y_preds = np.vstack((y_preds,y_pred))
                y_trues = np.vstack((y_trues,y_true))
                times = np.append(times,time)
                events = np.append(events,event)
            
            LOSS.append((epoch,step,loss.mean().item()))
            epoch_loss.append(loss.mean().item())

            # update data
            loop.set_description(f'Training... Epoch [{epoch+1}/{num_epoch}]')
            loop.set_postfix(batch_loss = loss.mean().item(),mean_loss = np.mean(np.array(epoch_loss)))

        scheduler.step()
        ##cal c-td
        c_td_train = cal_c_td(y_preds,y_trues,times,events)
        
        #### 
        data = []
        for i in range(len(LOSS)):
            data.append(LOSS[i][2])
        data = np.array(data)

        print('###Train data','C-td',round(c_td_train,6), 'mean loss:',np.mean(data[-train_size:]))

        test_loss_array = []
        if (epoch >= 0):
            model.eval()

            y_preds = 0
            y_trues = 0
            times = 0
            events = 0

            loop = tqdm(enumerate(test_loader), total =len(test_loader))
            for step, (batch_x,y_true,time,event) in loop:

                y_pred = model(adj,batch_x)

                loss1 = loss_func(y_true, y_pred).mean()  
                loss2 = rank_loss(y_true, y_pred,time,event)
                test_loss = loss1 + args.b*loss2

                test_loss_array.append(test_loss.mean().item())

                ### 计算c-index
                y_pred = y_pred.cpu().detach().numpy()
                y_true = y_true.cpu().detach().numpy()
                time = time.cpu().detach().numpy()
                event = event.cpu().detach().numpy()
                if(step == 0):
                    y_preds = y_pred
                    y_trues = y_true
                    times = time
                    events = event
                else:
                    y_preds = np.vstack((y_preds,y_pred))
                    y_trues = np.vstack((y_trues,y_true))
                    times = np.append(times,time)
                    events = np.append(events,event)

                loop.set_description('Testing ')
                loop.set_postfix(mean_loss = np.mean(np.array(test_loss_array)))
            
                

            c_td_test = cal_c_td(y_preds,y_trues,times,events)
            print('***Test data:','c_td:',round(c_td_test,6))
            TEST_max_C_TD = max(c_td_test,TEST_max_C_TD)
            print('***Max c_td:', round(TEST_max_C_TD,6))
  
    
    print('max c_td on test data is',TEST_max_C_TD)




def cal_c_td(y_pred,y_true,times,end):
    c_td = nnet_survival.get_time_c_td(y_pred,y_true,times,end)
    return c_td


if __name__ == '__main__':
    args = Parser(description='setting for training').args
    print(args)

    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, adj, node_nums, n_intervals,breaks,train_dataset,test_dataset = get_Data(device,args)

    hidden_dim = args.hidden_dim
    learining_rate = args.lr
    weight_decay_rate = args.weight_decay_rate
    feature_dim = args.feature_dim
    lr_ratio = args.lr_ratio


    loss_func = nnet_survival.surv_likelihood(n_intervals)
    rank_loss = nnet_survival.rank_loss

    model = models.Classifier(
        in_dim = args.feature_dim,
        hidden_dim =  hidden_dim,
        n_classes = n_intervals,
        node_nums = node_nums,
        args = args
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learining_rate, weight_decay=weight_decay_rate)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=lr_ratio)

    print('***start training...')
    train_and_eval(
        model,
        adj,
        loss_func,
        rank_loss,
        optimizer,
        scheduler,
        breaks,
        train_loader,
        test_loader,
        args,
        train_dataset,
        test_dataset
    )

    # print(args) 


    


