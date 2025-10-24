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
            tensors.append((file, data))

    for a, b in zip(tensors[:-1], tensors[1:]):
        print(a[0], "and", b[0])
        print("equal", torch.equal(a[1], b[1]))
        print("allclose", torch.allclose(a[1], b[1]))


if __name__ == "__main__":
    main()
