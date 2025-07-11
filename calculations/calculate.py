#!/usr/bin/env python
# vim: et:ts=4:sts=4:sw=4

# SPDX-License-Identifier: MIT
# Copyright 2025 The Alan Turing Institute

import torch
import struct
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import csv

class Calculations:
    size = 1024
    device = ""
    data_lines = []
    steps = []
    prefix = ""

    def __init__(self, size, scalefactor, seed, device, prefix):
        torch.manual_seed(seed)
        self.size = size
        self.device = device
        self.prefix = prefix
        self.scalefactor = scalefactor

    def seed(seed):
        torch.manual_seed(seed)

    # See https://docs.pytorch.org/xla/master/tutorials/precision_tutorial.html
    def binary_fraction_to_fp32(self, bstr: str) -> float:
        if bstr[:4] != "0b1.":
            raise ValueError(f"Invalid binary string: {bstr}")
        fraction_bits = bstr[4:]
        mantissa = 1.0
        for i, bit in enumerate(fraction_bits):
            mantissa += int(bit) * 2**-(i + 1)
        return float(mantissa)


    def fp32_to_binary_fraction(self, fp32_float: float) -> str:
        x_bytes = struct.pack(">f", fp32_float)  # Big-endian IEEE 754 float32
        as_int = struct.unpack(">I", x_bytes)[0]  # Interpret bits as uint32
        sign = (as_int >> 31) & 0b1
        exponent = (as_int >> 23) & 0xFF
        mantissa = as_int & 0x7FFFFF  # lower 23 bits
        return f"FORMAT:0b SIGN:{sign} EXPONENT:{exponent:08b} MANTISSA:{mantissa:023b} VALUE={fp32_float}"

    def random_check(self):
        print("Random number generation check start")
        for i in range(10):
            print(torch.rand((1, 1), dtype=torch.float64, device="cpu"))
        print("Random number generation check end")
        print()

    def get_rand_matrix(self, dtype):
        """Returns a diagonal matrix of shape (size, size), values between 0.999 and 1.111"""
        # Perform random generation on the CPU to avoid impacts from discrepancies betweeen GPUs
        rand_ = torch.rand((self.size, self.size), dtype=torch.float64, device="cpu") * 0.0002 + 0.0009
        result = rand_.to(self.device)
        return result

    # Perform matrix multiplication-and-adds at different precisions
    def matrix_multiple_mad(self, multiplier, accumulator, iterations, dtype):
        multiplier_cast = multiplier.type(dtype)
        accumulator_cast = accumulator.type(dtype)
        result = torch.eye(self.size, dtype=dtype, device=self.device)
        for count in range(iterations):
            result = torch.mm(result, multiplier_cast)
            result = torch.add(result, accumulator_cast)
        return result

    # Perform matrix multiplications at different precisions
    def matrix_multiple_mul(self, multiplier, iterations, dtype):
        multiplier_cast = multiplier.type(dtype)
        result = torch.eye(self.size, dtype=dtype, device=self.device)
        for count in range(iterations):
            result = torch.mm(result, multiplier_cast)
        return result

    # Perform matrix multiplication-and-adds at different precisions
    def matrix_multiple_add(self, accumulator, iterations, dtype):
        accumulator_cast = accumulator.type(dtype)
        result = torch.zeros(self.size, dtype=dtype, device=self.device)
        for count in range(iterations):
            result = torch.add(result, accumulator_cast)
        return result

    def matrix_compare(self, mat1, mat2):
        mat1_64 = mat1.detach().clone()
        mat1_64.type(torch.float64)
        mat2_64 = mat2.detach().clone()
        mat2_64.type(torch.float64)
        mse = nn.MSELoss()(mat1_64, mat2_64)
        return mse

    def plot_graph(self, filename):
        print("Plotting graph: {}".format(filename))
        fig, ax = plt.subplots(figsize=(8,5))
        ax.set_yscale("log")
        for pos, (name, data) in enumerate(self.data_lines):
            if pos % 4:
                linestyle = ['-', ':'][(pos // 4) % 2]
                marker = [7, 6][(pos // 4) % 2]
                linewidth = [0.5, 1.0][(pos // 4) % 2]
                colour = ['#9cd839', '#4dc169', '#228b8b', '#471b6d'][pos % 4]
                ax.plot(self.steps, data, color=colour, linestyle=linestyle, marker=marker, label=name, linewidth=linewidth)
        ax.set_xlabel("Matrix Multiply-and-Accumulate steps")
        ax.set_ylabel("Cumulative Mean Square Error")
        ax.legend();
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    def load_data(self, filename):
        with open(filename, 'r') as fh:
            self.steps = []
            reader = csv.reader(fh)
            headers = next(reader)
            lines = [[] for _ in range(len(headers))]
            for row in reader:
                self.steps.append(int(row[0]))
                for pos in range(1, len(row)):
                    lines[pos].append(float(row[pos]))
            for pos in range(1, len(lines)):
                self.data_lines.append((headers[pos], lines[pos]))

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

    def generate_data_mad(self, filename):
        steps_list = []
        mse_f64_list = []
        mse_f32_list = []
        mse_f16_list = []
        mse_bf16_list = []
        self.reset_datalines()
        with open(filename, "w") as fh:
            header = ",".join(["\"{}{}\"".format(self.prefix, name) for name in ["fp64", "fp32", "fp16", "bf16"]])
            fh.write("\"Steps\",{}\n".format(header))
            multiplier = calc.get_rand_matrix(torch.float64)
            accumulator = calc.get_rand_matrix(torch.float64)
            for steps in range(16 * self.scalefactor, 513 * self.scalefactor, 16 * self.scalefactor):
                result_f64 = calc.matrix_multiple_mad(multiplier, accumulator, steps, torch.float64)
                result_f32 = calc.matrix_multiple_mad(multiplier, accumulator, steps, torch.float32)
                result_f16 = calc.matrix_multiple_mad(multiplier, accumulator, steps, torch.float16)
                result_bf16 = calc.matrix_multiple_mad(multiplier, accumulator, steps, torch.bfloat16)

                mse_f64 = calc.matrix_compare(result_f64, result_f64)
                mse_f32 = calc.matrix_compare(result_f64, result_f32)
                mse_f16 = calc.matrix_compare(result_f64, result_f16)
                mse_bf16 = calc.matrix_compare(result_f64, result_bf16)

                steps_list.append(steps)
                mse_f64_list.append(mse_f64.cpu())
                mse_f32_list.append(mse_f32.cpu())
                mse_f16_list.append(mse_f16.cpu())
                mse_bf16_list.append(mse_bf16.cpu())

                fh.write(("{},{},{},{},{}\n".format(steps, mse_f64, mse_f32, mse_f16, mse_bf16)))

        self.steps = steps_list
        self.data_lines.append(("{}fp64".format(self.prefix), mse_f64_list))
        self.data_lines.append(("{}fp32".format(self.prefix), mse_f32_list))
        self.data_lines.append(("{}fp16".format(self.prefix), mse_f16_list))
        self.data_lines.append(("{}bf16".format(self.prefix), mse_bf16_list))

    def generate_data_mul(self, filename):
        steps_list = []
        mse_f64_list = []
        mse_f32_list = []
        mse_f16_list = []
        mse_bf16_list = []
        self.reset_datalines()
        with open(filename, "w") as fh:
            header = ",".join(["\"{}{}\"".format(self.prefix, name) for name in ["fp64", "fp32", "fp16", "bf16"]])
            fh.write("\"Steps\",{}\n".format(header))
            multiplier = calc.get_rand_matrix(torch.float64)
            for steps in range(16 * self.scalefactor, 513 * self.scalefactor, 16 * self.scalefactor):
                result_f64 = calc.matrix_multiple_mul(multiplier, steps, torch.float64)
                result_f32 = calc.matrix_multiple_mul(multiplier, steps, torch.float32)
                result_f16 = calc.matrix_multiple_mul(multiplier, steps, torch.float16)
                result_bf16 = calc.matrix_multiple_mul(multiplier, steps, torch.bfloat16)

                mse_f64 = calc.matrix_compare(result_f64, result_f64)
                mse_f32 = calc.matrix_compare(result_f64, result_f32)
                mse_f16 = calc.matrix_compare(result_f64, result_f16)
                mse_bf16 = calc.matrix_compare(result_f64, result_bf16)

                steps_list.append(steps)
                mse_f64_list.append(mse_f64.cpu())
                mse_f32_list.append(mse_f32.cpu())
                mse_f16_list.append(mse_f16.cpu())
                mse_bf16_list.append(mse_bf16.cpu())

                fh.write(("{},{},{},{},{}\n".format(steps, mse_f64, mse_f32, mse_f16, mse_bf16)))

        self.steps = steps_list
        self.data_lines.append(("{}fp64".format(self.prefix), mse_f64_list))
        self.data_lines.append(("{}fp32".format(self.prefix), mse_f32_list))
        self.data_lines.append(("{}fp16".format(self.prefix), mse_f16_list))
        self.data_lines.append(("{}bf16".format(self.prefix), mse_bf16_list))

    def generate_data_add(self, filename):
        steps_list = []
        mse_f64_list = []
        mse_f32_list = []
        mse_f16_list = []
        mse_bf16_list = []
        self.reset_datalines()
        with open(filename, "w") as fh:
            header = ",".join(["\"{}{}\"".format(self.prefix, name) for name in ["fp64", "fp32", "fp16", "bf16"]])
            fh.write("\"Steps\",{}\n".format(header))
            accumulator = calc.get_rand_matrix(torch.float64)
            for steps in range(16 * self.scalefactor, 513 * self.scalefactor, 16 * self.scalefactor):
                result_f64 = calc.matrix_multiple_add(accumulator, steps, torch.float64)
                result_f32 = calc.matrix_multiple_add(accumulator, steps, torch.float32)
                result_f16 = calc.matrix_multiple_add(accumulator, steps, torch.float16)
                result_bf16 = calc.matrix_multiple_add(accumulator, steps, torch.bfloat16)

                mse_f64 = calc.matrix_compare(result_f64, result_f64)
                mse_f32 = calc.matrix_compare(result_f64, result_f32)
                mse_f16 = calc.matrix_compare(result_f64, result_f16)
                mse_bf16 = calc.matrix_compare(result_f64, result_bf16)

                steps_list.append(steps)
                mse_f64_list.append(mse_f64.cpu())
                mse_f32_list.append(mse_f32.cpu())
                mse_f16_list.append(mse_f16.cpu())
                mse_bf16_list.append(mse_bf16.cpu())

                fh.write(("{},{},{},{},{}\n".format(steps, mse_f64, mse_f32, mse_f16, mse_bf16)))

        self.steps = steps_list
        self.data_lines.append(("{}fp64".format(self.prefix), mse_f64_list))
        self.data_lines.append(("{}fp32".format(self.prefix), mse_f32_list))
        self.data_lines.append(("{}fp16".format(self.prefix), mse_f16_list))
        self.data_lines.append(("{}bf16".format(self.prefix), mse_bf16_list))

    def generate_data_mad_fuzz(self, filename):
        steps_list = []
        mse_f64_list = []
        mse_f32_list = []
        mse_f16_list = []
        mse_bf16_list = []
        self.reset_datalines()
        with open(filename, "w") as fh:
            header = ",".join(["\"{}{}\"".format(self.prefix, name) for name in ["fp64", "fp32", "fp16", "bf16"]])
            fh.write("\"Steps\",{}\n".format(header))
            for steps in range(16 * self.scalefactor, 513 * self.scalefactor, 16 * self.scalefactor):
                multiplier = calc.get_rand_matrix(torch.float64)
                print(multiplier)
                accumulator = calc.get_rand_matrix(torch.float64)
                result_f64 = calc.matrix_multiple_mad(multiplier, accumulator, steps, torch.float64)
                result_f32 = calc.matrix_multiple_mad(multiplier, accumulator, steps, torch.float32)
                result_f16 = calc.matrix_multiple_mad(multiplier, accumulator, steps, torch.float16)
                result_bf16 = calc.matrix_multiple_mad(multiplier, accumulator, steps, torch.bfloat16)

                mse_f64 = calc.matrix_compare(result_f64, result_f64)
                mse_f32 = calc.matrix_compare(result_f64, result_f32)
                mse_f16 = calc.matrix_compare(result_f64, result_f16)
                mse_bf16 = calc.matrix_compare(result_f64, result_bf16)

                steps_list.append(steps)
                mse_f64_list.append(mse_f64.cpu())
                mse_f32_list.append(mse_f32.cpu())
                mse_f16_list.append(mse_f16.cpu())
                mse_bf16_list.append(mse_bf16.cpu())

                fh.write(("{},{},{},{},{}\n".format(steps, mse_f64, mse_f32, mse_f16, mse_bf16)))

        self.steps = steps_list
        self.data_lines.append(("{}fp64".format(self.prefix), mse_f64_list))
        self.data_lines.append(("{}fp32".format(self.prefix), mse_f32_list))
        self.data_lines.append(("{}fp16".format(self.prefix), mse_f16_list))
        self.data_lines.append(("{}bf16".format(self.prefix), mse_bf16_list))

parser = argparse.ArgumentParser()
parser.add_argument("--accelerator", "-a", type=str, default="auto", choices=["auto", "cpu", "cuda", "xpu"], help="Accelerator to use")
parser.add_argument("--prefix", "-p", type=str, default="", help="Graph line name prefix")
parser.add_argument("--filein", "-i", type=str, default="", help="Filename for the additional data to load")
parser.add_argument("--fileout", "-o", type=str, default="", help="Filename to output the data to")
args = parser.parse_args()

calc = Calculations(3, 10, 42, args.accelerator, args.prefix)
calc.random_check()
if args.fileout:
    fileout = calc.suffix_path(args.fileout, "-mad-3x3")
    calc.generate_data_mad(fileout)
if args.filein:
    filein = calc.suffix_path(args.filein, "-mad-3x3")
    calc.load_data(filein)
calc.plot_graph("plot-error-mad-3x3.png")
calc.plot_graph("plot-error-mad-3x3.pdf")

calc = Calculations(1024, 1, 42, args.accelerator, args.prefix)
if args.fileout:
    fileout = calc.suffix_path(args.fileout, "-mad-1024x1024")
    calc.generate_data_mad(fileout)
if args.filein:
    filein = calc.suffix_path(args.filein, "-mad-1024x1024")
    calc.load_data(filein)
calc.plot_graph("plot-error-mad-1024x1024.png")
calc.plot_graph("plot-error-mad-1024x1024.pdf")

calc = Calculations(1024, 1, 42, args.accelerator, args.prefix)
if args.fileout:
    fileout = calc.suffix_path(args.fileout, "-mul-1024x1024")
    calc.generate_data_mul(fileout)
if args.filein:
    filein = calc.suffix_path(args.filein, "-mul-1024x1024")
    calc.load_data(filein)
calc.plot_graph("plot-error-mul-1024x1024.png")
calc.plot_graph("plot-error-mul-1024x1024.pdf")

calc = Calculations(1024, 1, 42, args.accelerator, args.prefix)
if args.fileout:
    fileout = calc.suffix_path(args.fileout, "-add-1024x1024")
    calc.generate_data_add(fileout)
if args.filein:
    filein = calc.suffix_path(args.filein, "-add-1024x1024")
    calc.load_data(filein)
calc.plot_graph("plot-error-add-1024x1024.png")
calc.plot_graph("plot-error-add-1024x1024.pdf")

calc = Calculations(1024, 1, 42, args.accelerator, args.prefix)
if args.fileout:
    fileout = calc.suffix_path(args.fileout, "-mad-fuzz-1024x1024")
    calc.generate_data_mad_fuzz(fileout)
if args.filein:
    filein = calc.suffix_path(args.filein, "-mad-fuzz-1024x1024")
    calc.load_data(filein)
calc.plot_graph("plot-error-mad-fuzz-1024x1024.png")
calc.plot_graph("plot-error-mad-fuzz-1024x1024.pdf")

if args.fileout:
    for seed in range(42, 52):
        print("Seed: {}".format(seed))
        calc = Calculations(1024, 1, seed, args.accelerator, args.prefix)
        fileout = calc.index_path(args.fileout, seed)
        calc.generate_data_mad(fileout)
        if args.filein:
            filein = calc.index_path(args.filein, seed)
            calc.load_data(filein)
        calc.plot_graph("plot-error-mad-{:04}.png".format(seed))
        calc.plot_graph("plot-error-mad-{:04}.pdf".format(seed))

