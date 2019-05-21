from n2v_flim import n2v_flim
import os

root = '/home/seawar/data/2019-05-08/projects/'

sets = ['I=0010', 'I=0020', 'I=0030', 'I=0050', 'I=0080', 'I=0130', 'I=0220', 'I=0360', 'I=0600', 'I=1000']
#sets = ['I=0600']

for s in sets[0:2]:
   n2v_flim(root + s + '.flimfit', 64)

