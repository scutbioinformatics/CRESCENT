

import numpy as np
import torch
import torch.nn.functional as F
from numba import jit

##### code come from nnet_survival ####
##### code come from nnet_survival ####

def surv_likelihood(n_intervals):
  """Create custom Keras loss function for neural network survival model. 
  Arguments
      n_intervals: the number of survival time intervals
  Returns
      Custom loss function that can be used with Keras
  """
  def loss(y_true, y_pred):
    """
    Required to have only 2 arguments by Keras.
    Arguments
        y_true: Tensor.
          First half of the values is 1 if individual survived that interval, 0 if not.
          Second half of the values is for individuals who failed, and is 1 for time interval during which failure occured, 0 for other intervals.
          See make_surv_array function.
        y_pred: Tensor, predicted survival probability (1-hazard probability) for each time interval.
    Returns
        Vector of losses for this minibatch.
    """
    cens_uncens = 1. + y_true[:,0:n_intervals] * (y_pred-1.) #component for all individuals
    uncens = 1. - y_true[:,n_intervals:2*n_intervals] * y_pred #component for only uncensored individuals
    eps = torch.finfo(torch.float32).eps
    return torch.sum(-torch.log(torch.clamp(torch.cat((cens_uncens,uncens),1),eps,None)),axis=-1) #return -log likelihood
  return loss


def make_surv_array(t,f,breaks):
  """Transforms censored survival data into vector format that can be used in Keras.
    Arguments
        t: Array of failure/censoring times.
        f: Censoring indicator. 1 if failed, 0 if censored.
        breaks: Locations of breaks between time intervals for discrete-time survival model (always includes 0)
    Returns
        Two-dimensional array of survival data, dimensions are number of individuals X number of time intervals*2
  """
  n_samples=t.shape[0]
  n_intervals=len(breaks)-1
  timegap = breaks[1:] - breaks[:-1]
  breaks_midpoint = breaks[:-1] + 0.5*timegap
  y_train = np.zeros((n_samples,n_intervals*2))
  for i in range(n_samples):
    if f[i]: #if failed (not censored)
      y_train[i,0:n_intervals] = 1.0*(t[i]>=breaks[1:]) #give credit for surviving each time interval where failure time >= upper limit
      if t[i]<breaks[-1]: #if failure time is greater than end of last time interval, no time interval will have failure marked
        y_train[i,n_intervals+np.where(t[i]<breaks[1:])[0][0]]=1 #mark failure at first bin where survival time < upper break-point
    else: #if censored
      y_train[i,0:n_intervals] = 1.0*(t[i]>=breaks_midpoint) #if censored and lived more than half-way through interval, give credit for surviving the interval.
  return y_train

def nnet_pred_surv(y_pred, breaks, fu_time):
#Predicted survival probability from Nnet-survival model
#Inputs are Numpy arrays.
#y_pred: Rectangular array, each individual's conditional probability of surviving each time interval
#breaks: Break-points for time intervals used for Nnet-survival model, starting with 0
#fu_time: Follow-up time point at which predictions are needed
#
#Returns: predicted survival probability for each individual at specified follow-up time
  y_pred=np.cumprod(y_pred, axis=1)
  pred_surv = []
  for i in range(y_pred.shape[0]):
    pred_surv.append(np.interp(fu_time,breaks[1:],y_pred[i,:]))
  return np.array(pred_surv)

##### code come from nnet_survival ####
##### code come from nnet_survival ####


def rank_loss(y_true, y_pred,time,event):
    t = time.unsqueeze(1).float()
    k = event
    mask = y_true[:,:19] + y_true[:,19:] 

    sigma1 = 0.1

    one_vector = torch.ones_like(t, dtype = torch.float32)
    I_2 = torch.eq(k, 1).float()
    I_2 = torch.diag_embed(torch.squeeze(I_2))

    tmp_e = 1. - y_pred

    R = torch.mm(tmp_e, mask.t()) 

    diag_R = torch.reshape(torch.diagonal(R), (-1, 1)) 
    R = torch.mm(one_vector, diag_R.t()) - R 
    R = R.t() 


    T = F.relu(torch.sign(torch.mm(one_vector, t.t()) - torch.mm(t,one_vector.t())))
    T = torch.mm(I_2, T)
    temp_eta = torch.mean(T * torch.exp(-R/sigma1), 1, keepdim = True)

    return temp_eta.sum()


@jit(nopython=True)
def get_time_c_td(y_pred,y_true,times,end):
    '''
    y_pred : our model output
    times
    end:dead =1 cencored 0
    breaks:y_pred的时间间隔
    '''
    mask = y_true[:,:19] + y_true[:,19:]
     
    pair_num = 0
    correct_pair_unm = 0
    tied_pair_num = 0
    n_sample = y_pred.shape[0]
    
    for i in range(n_sample):
        for j in range(i+1,n_sample):
            if ( (end[i] == 1) and (end[j] == 1) ) or (end[i] == 1 and times[i] <= times[j]) or (end[j] == 1 and times[j] <= times[i]):
                if end[i] == 1 and end[j] == 1 and times[i] == times[j]:
                    continue

                time_i = times[i] if end[i] == 1 else 1e9
                time_j = times[j] if end[j] == 1 else 1e9
                min_time_index = i if time_i < time_j else j

                surv_i = np.sum(y_pred[i] * mask[min_time_index])
                surv_j = np.sum(y_pred[j] * mask[min_time_index])

                pair_num = pair_num + 1
                if (time_i > time_j and surv_i > surv_j) or (time_j > time_i and surv_j > surv_i):
                    correct_pair_unm = correct_pair_unm + 1
    
    c_index = (correct_pair_unm ) / pair_num
    return c_index