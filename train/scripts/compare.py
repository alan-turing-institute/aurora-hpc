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
            tensors.append(data)

    for a, b in zip(tensors[:-1], tensors[1:]):
        print(torch.equal(a, b))
        print(torch.allclose(a, b))


if __name__ == "__main__":
    main()
