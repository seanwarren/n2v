from n2v_flim import n2v_flim
import sys

if len(sys.argv[0]) < 2:
    print("Missing input folder name")
    sys.exit(1)

root = sys.argv[1]
n2v_flim(root, 64)