import numpy as np
import h5py
import re

from csbdeep.models import Config, CARE
import numpy as np
import shutil
import os
from csbdeep.utils import plot_some, plot_history
from csbdeep.utils.n2v_utils import manipulate_val_data

# We need to normalize the data before we feed it into our network, and denormalize it afterwards.
def normalize(img, mean, std):
    zero_mean = img - mean
    return zero_mean/std

def denormalize(x, mean, std):
    return x*std + mean

def sorted_nicely( l ):
    """ Sorts the given iterable in the way that is expected.

    Required arguments:
    l -- The iterable to be sorted.

    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)

def param(group, name):
   return 'G' + str(group) + '_' + name

def extract_results(filename):
   """ Extract intensity data from a FLIMfit results file.
   Converts any fraction data (e.g. beta, gamma) to contributions

   Required arguments:
   filename - the name of the file to load 
   """
   file = h5py.File(filename,'r') 
   results = file['results']

   keys = sorted_nicely(results.keys())
   params = sorted_nicely(results['image 1'].keys())

   groups = []

   g = 1
   while(param(g,'I_0') in params):
      group = [param(g,'I_0')]

      name_search = [param(g,'gamma'), param(g,'beta')]

      for name in name_search:
         if len(group) == 1:
            group = group + [x for x in params if x.startswith(name)]

      groups.append(group)
      g = g + 1

   print(groups)

   X = []
   mask = []
   for k in keys:
      A = []
      m = np.array([False])

      for group in groups:
         I_0 = results[k][group[0]]
         m = m | ~np.isfinite(I_0)
         if len(group) == 1:
            A.append(I_0)
         else:
            for i in range(1,len(group)):
               A.append(results[k][group[i]][()] * I_0)

      A = np.stack(A, axis=-1)
      A[np.isnan(A)] = 0
      X.append(A)
      mask.append(m)
      
   X = np.stack(X)
   mask = np.stack(mask)

   return X, groups, mask


def insert_results(filename, X, groups):

   file = h5py.File(filename,'a') 
   results = file['results']

   # Denoise all images
   for i in range(X.shape[0]):
      key = 'image ' + str(i+1)

      idx = 0
      for group in groups:
         num_image = max(1, len(group)-1)
         A = X[i,:,:,slice(idx,idx+num_image+1)]
         if len(group) > 1:
            I_0 = np.sum(A, axis=-1, keepdims=True)
            A = A / I_0
            A = np.concatenate((I_0, A), axis=2)
         for j in range(len(group)):
            results[key][group[j]].write_direct(np.ascontiguousarray(A[:,:,j]))

         idx = idx + num_image


def n2v_flim(project, n2v_num_pix=32):
   
   results_file = os.path.join(project, 'fit_results.hdf5')

   X, groups, mask = extract_results(results_file)
   data_shape = np.shape(X)
   print(data_shape)

   mean, std = np.mean(X), np.std(X)
   X = normalize(X, mean, std)

   X_val = X[0:10,...]

   # We concatenate an extra channel filled with zeros. It will be internally used for the masking.
   Y = np.concatenate((X, np.zeros(X.shape)), axis=-1)
   Y_val = np.concatenate((X_val.copy(), np.zeros(X_val.shape)), axis=-1) 

   n_x = X.shape[1]
   n_chan = X.shape[-1]

   manipulate_val_data(X_val, Y_val, num_pix=n_x*n_x*2/n2v_num_pix , shape=(n_x, n_x))


   # You can increase "train_steps_per_epoch" to get even better results at the price of longer computation. 
   config = Config('SYXC', 
                  n_channel_in=n_chan, 
                  n_channel_out=n_chan, 
                  unet_kern_size = 5, 
                  unet_n_depth = 2,
                  train_steps_per_epoch=200, 
                  train_loss='mae',
                  train_epochs=35,
                  batch_norm = False, 
                  train_scheme = 'Noise2Void', 
                  train_batch_size = 128, 
                  n2v_num_pix = n2v_num_pix,
                  n2v_patch_shape = (n2v_num_pix, n2v_num_pix), 
                  n2v_manipulator = 'uniform_withCP', 
                  n2v_neighborhood_radius='5')

   vars(config)

   model = CARE(config, 'n2v_model', basedir=project)

   history = model.train(X, Y, validation_data=(X_val,Y_val))

   model.load_weights(name='weights_best.h5')

   output_project = project.replace('.flimfit','-n2v.flimfit')
   if os.path.exists(output_project) : shutil.rmtree(output_project)
   shutil.copytree(project, output_project)

   output_file = os.path.join(output_project, 'fit_results.hdf5')

   X_pred = np.zeros(X.shape)
   for i in range(X.shape[0]):
      X_pred[i,...] = denormalize(model.predict(X[i], axes='YXC',normalizer=None), mean, std)

   X_pred[mask] = np.NaN

   insert_results(output_file, X_pred, groups)
