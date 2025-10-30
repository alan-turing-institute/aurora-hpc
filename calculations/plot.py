#!/usr/bin/env python
# vim: et:ts=4:sts=4:sw=4

# SPDX-License-Identifier: MIT
# Copyright 2025 The Alan Turing Institute

import struct
import matplotlib.pyplot as plt
import argparse
import csv

class Plot:
    size = 1024
    data_lines = []
    steps = []

    def __init__(self, operation):
        self.operation = operation

    def plot_graph(self, filename):
        print("Plotting graph: {}".format(filename))
        fig, ax = plt.subplots(figsize=(8,5))
        ax.set_yscale("log")
        for pos, (name, data) in enumerate(self.data_lines):
            device, datatype = name.replace(" (", "-(").split()
            if datatype != "fp64":
                linestyle, marker, linewidth = {
                    "CUDA": ("-", 7, 0.5),
                    "XPU": ("-", 6, 0.5),
                    "CPU": (":", "x", 1.0),
                    "CPU-(Baskerville)": (":", "x", 1.0),
                    "CPU-(DAWN)": (":", "+", 1.0),
                }[device]
                colour = {
                    "fp64": "#9cd839",
                    "fp32": "#4dc169",
                    "fp16": "#228b8b",
                    "bf16": "#471b6d",
                }[datatype]
                ax.plot(self.steps, data, color=colour, linestyle=linestyle, marker=marker, label=name, linewidth=linewidth)
        ax.set_xlabel("Matrix {} steps".format(self.operation))
        ax.set_ylabel("Cumulative Mean Square Error")
        ax.legend();
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    def load_data(self, filename):
        with open(filename, 'r') as fh:
            steps = []
            reader = csv.reader(fh)
            headers = next(reader)
            lines = [[] for _ in range(len(headers))]
            for row in reader:
                steps.append(int(row[0]))
                for pos in range(1, len(row)):
                    lines[pos].append(float(row[pos]))
            if (self.steps == []) or (steps == self.steps):
                self.steps = steps
                for pos in range(1, len(lines)):
                    self.data_lines.append((headers[pos], lines[pos]))
            else:
                print("Error: can't combine data with different x-axes")

    def index_path(self, filename, seed):
        pos = filename.rfind(".")
        if pos < 0:
            pos = len(filename)
        return "{}-{:04}{}".format(filename[:pos], seed, filename[pos:])

    def suffix_path(self, filename, suffix):
        pos = filename.rfind(".")
        if pos < 0:
            pos = len(filename)
        return "{}{}{}".format(filename[:pos], suffix, filename[pos:])

    def reset_datalines(self):
        self.data_lines.clear()
        self.steps.clear()

parser = argparse.ArgumentParser()
parser.add_argument("--operation", "-x", default="Multpiply-and-Accumulate", help="The name of the operation being plotted")
parser.add_argument("--filein", "-i", action="append", help="Filename for the additional data to load")
parser.add_argument("--fileout", "-o", help="Filename to save the graphs out to, without extension")
args = parser.parse_args()

plot = Plot(args.operation)
for filein in args.filein:
    plot.load_data(filein)
plot.plot_graph("plot-error-{}.png".format(args.fileout))
plot.plot_graph("plot-error-{}.pdf".format(args.fileout))
