import os, re, sys
from n2v_flim import n2v_flim

offset = int(sys.argv[1])
n = int(sys.argv[2])

os.environ["CUDA_VISIBLE_DEVICES"]=str(offset)

root = '/home/seawar/data/2019-05-22/projects/'

# get only unprocessed projects
sets = os.listdir(root)
regex = re.compile(r'I=\d+\.flimfit')
sets = list(filter(regex.search, sets))

for i in range(offset,len(sets),n):
    print("Processing: " + sets[i])
    n2v_flim(root + sets[i], 64)

