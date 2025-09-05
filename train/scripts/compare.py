import pathlib
import pickle
import sys

import torch


def main():
    mydir = pathlib.Path(sys.argv[1])
    print(mydir)
    tensors = []
    for file in mydir.iterdir():
        print("opening ", file)
        with open(file, "rb") as f:
            data = pickle.load(f)
            tensors.append((data, file))

    for a, b in zip(tensors[:-1], tensors[1:]):
        print(a[1], b[1])
        print(torch.equal(a[0].surf_vars["10v"], b[0].surf_vars["10v"]))
        print(torch.allclose(a[0].surf_vars["10v"], b[0].surf_vars["10v"]))


if __name__ == "__main__":
    main()
