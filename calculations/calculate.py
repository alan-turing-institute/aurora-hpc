#!/usr/bin/env python
# vim: et:ts=4:sts=4:sw=4

# SPDX-License-Identifier: MIT
# Copyright 2025 The Alan Turing Institute

import torch
import torch.nn as nn
import argparse

class Calculations:
    """A class for performing calculation error comparisons."""
    size = 1024
    device = ""
    data_lines = []
    steps = []
    prefix = ""
    precisions = [torch.float64, torch.float32, torch.float16, torch.bfloat16]
    precision_names = {
        torch.float64: "fp64",
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
    }
    device_precisions = {
        "cpu": [torch.float64, torch.float32]
    }

    def __init__(self, size, scalefactor, seed, device, prefix):
        """Initialise the calculations, including setting the random seed.

        Consequently a new instance should be created for each calculation run.
        """
        self.seed = seed
        torch.manual_seed(self.seed)
        self.size = size
        self.device = device
        self.prefix = prefix
        self.scalefactor = scalefactor
        if self.device in self.device_precisions:
            self.precisions = self.device_precisions[self.device]

    def random_check(self):
        """Print a list of 10 random numbers to the console.

        This can be used a check to ensure identical random numbers are being generated on all devices.
        """
        print("Random number generation check start")
        for i in range(10):
            value = torch.rand((1, 1), dtype=torch.float64, device="cpu").item()
            print("  {}".format(format(value, '.60g')))
        print("Random number generation check end")
        print()

    def get_rand_matrix(self, dtype, scale, offset):
        """Returns a diagonal matrix of shape (size, size), values between 0.999 and 1.111"""
        # Perform random generation on the CPU to avoid impacts from discrepancies betweeen GPUs
        result_cpu = (2.0 * (torch.rand((self.size, self.size), dtype=torch.float64, device="cpu") - 1) * scale) + offset
        result_device = result_cpu.detach().clone()
        result_device = result_device.to(self.device)
        #value = result[0,0].item()
        return result_cpu, result_device

    def matrix_multiple_mad(self, multiplier, accumulator, iterations, dtype, device):
        """Perform matrix multiplication-and-adds at different precisions"""
        multiplier_cast = multiplier.type(dtype)
        accumulator_cast = accumulator.type(dtype)
        result = torch.eye(self.size, dtype=dtype, device=device)
        for count in range(iterations):
            result = torch.mm(result, multiplier_cast)
            result = torch.add(result, accumulator_cast)
        return result

    def matrix_multiple_mul(self, multiplier, _, iterations, dtype, device):
        """Perform matrix multiplications at different precisions"""
        multiplier_cast = multiplier.type(dtype)
        result = torch.eye(self.size, dtype=dtype, device=device)
        for count in range(iterations):
            result = torch.mm(result, multiplier_cast)
        return result

    def matrix_multiple_add(self, _, accumulator, iterations, dtype, device):
        """Perform matrix multiplication-and-adds at different precisions"""
        accumulator_cast = accumulator.type(dtype)
        result = torch.zeros(self.size, dtype=dtype, device=device)
        for count in range(iterations):
            result = torch.add(result, accumulator_cast)
        return result

    def matrix_compare(self, mat1, mat2):
        """Compare two matrices with 64-bit precision.

        Returns: the Mean Square Error of the matrix elementwise.
        """
        mat1_64 = mat1.detach().clone()
        mat1_64.type(torch.float64)
        mat2_64 = mat2.detach().clone()
        mat2_64.type(torch.float64)
        mse = nn.MSELoss()(mat1_64, mat2_64)
        return mse

    def save_data(self, filename):
        """Exports collected data to file.

        Exports the accumulated dataset to file using the given filename.

        Arguments:
            filename: the filename to export to.
        """
        with open(filename, 'w') as fh:
            header = ",".join(["\"{}{}\"".format(self.prefix, name) for name in [self.precision_names[precision] for precision in self.precisions]])
            fh.write("\"Steps\",{}\n".format(header))

            for pos in range(len(self.steps)):
                step = self.steps[pos]
                values = ",".join([format(data_line[1][pos].item(), ".10e") for data_line in self.data_lines])
                fh.write(("{},{}\n".format(step, values)))

    def index_path(self, filename, seed):
        """Creates a file name based on a seed value.

        Takes an existing filename and adds the seed value just before the extension.

        Returns: a filename that includes the seed value.
        """
        pos = filename.rfind(".")
        if pos < 0:
            pos = len(filename)
        return "{}-{:04}{}".format(filename[:pos], seed, filename[pos:])

    def suffix_path(self, filename, suffix):
        """Creates a file name based on a provided string suffix.

        Takes an existing filename and adds the suffix string just before the extension.

        Returns: a filename that includes the string suffix.
        """
        pos = filename.rfind(".")
        if pos < 0:
            pos = len(filename)
        return "{}{}{}".format(filename[:pos], suffix, filename[pos:])

    def reset_datalines(self):
        """Clears out the collected data from the class."""
        self.data_lines.clear()
        self.steps.clear()

    def generate_data(self, filename, matrix_operation, fuzz, name):
        """Perform error calculations using the provided function.

        Arguments:
            filename: the file to write the results out to.
            matrix_operation: the function to use each time round the loop.
            fuzz: True for new random matrices to be generated each time round the loop, False o/w.
            name: The name to print to the screen to gauge progress.
        """
        print("Error calculations for {} x {} matrix with seed {} using {} on {}".format(self.size, self.size, self.seed, name, self.prefix))
        steps_list = []
        self.reset_datalines()
        mse_lists = [[] for _ in range(len(self.precisions))]

        multiplier_cpu, multiplier_device = self.get_rand_matrix(torch.float64, 0.1 / self.size, 1.0 / self.size)
        accumulator_cpu, accumulator_device = self.get_rand_matrix(torch.float64, 0.1, 0.0)
        for steps in range(16 * self.scalefactor, 513 * self.scalefactor, 16 * self.scalefactor):
            if fuzz:
                multiplier_cpu, multiplier_device = self.get_rand_matrix(torch.float64, 0.1 / self.size, 1.0 / self.size)
                accumulator_cpu, accumulator_device = self.get_rand_matrix(torch.float64, 0.1, 0.0)
            steps_list.append(steps)
            result_cpu = matrix_operation(multiplier_cpu, accumulator_cpu, steps, torch.float64, "cpu")
            for pos, precision in enumerate(self.precisions):
                result = matrix_operation(multiplier_device, accumulator_device, steps, precision, self.device)
                mse = self.matrix_compare(result_cpu, result.cpu())
                mse_lists[pos].append(mse.cpu())

        self.steps = steps_list
        for pos, precision in enumerate(self.precisions):
            mse_lists[pos] = mse_lists[pos]
            self.data_lines.append(("{}{}".format(self.prefix, self.precision_names[precision]), mse_lists[pos]))

        self.save_data(filename)

    def generate_data_mad(self, filename):
        """Perform error calculations using matrix multiply-and-add."""
        self.generate_data(filename, self.matrix_multiple_mad, False, "mad")

    def generate_data_mul(self, filename):
        """Perform error calculations using matrux multiply."""
        self.generate_data(filename, self.matrix_multiple_mul, False, "mul")

    def generate_data_add(self, filename):
        """Perform error calculations using matrix add."""
        self.generate_data(filename, self.matrix_multiple_add, False, "add")

    def generate_data_mad_fuzz(self, filename):
        """Perform error calculations using matrix multiply-and-add with fuzzing.

        Fuzzing in this case means that a different matrix multipier and accumulator
        are generated at each step, rather then re-using the same pair of matrices
        each time round the loop.
        """
        self.generate_data(filename, self.matrix_multiple_mad, True, "mad fuzz")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accelerator", "-a", type=str, default="auto", choices=["auto", "cpu", "cuda", "xpu"], help="Accelerator to use")
    parser.add_argument("--prefix", "-p", type=str, default="", help="Graph line name prefix")
    parser.add_argument("--fileout", "-o", type=str, required=True, help="Filename to output the data to")
    args = parser.parse_args()

    calc = Calculations(3, 10, 42, args.accelerator, args.prefix)
    calc.random_check()

    calc = Calculations(3, 10, 42, args.accelerator, args.prefix)
    fileout = calc.suffix_path(args.fileout, "-mad-3x3")
    calc.generate_data_mad(fileout)

    calc = Calculations(1024, 1, 42, args.accelerator, args.prefix)
    fileout = calc.suffix_path(args.fileout, "-mad-1024x1024")
    calc.generate_data_mad(fileout)

    calc = Calculations(1024, 1, 42, args.accelerator, args.prefix)
    fileout = calc.suffix_path(args.fileout, "-mul-1024x1024")
    calc.generate_data_mul(fileout)

    calc = Calculations(1024, 1, 42, args.accelerator, args.prefix)
    fileout = calc.suffix_path(args.fileout, "-add-1024x1024")
    calc.generate_data_add(fileout)

    calc = Calculations(1024, 1, 42, args.accelerator, args.prefix)
    fileout = calc.suffix_path(args.fileout, "-mad-fuzz-1024x1024")
    calc.generate_data_mad_fuzz(fileout)

    for seed in range(42, 52):
        calc = Calculations(1024, 1, seed, args.accelerator, args.prefix)
        fileout = calc.index_path(args.fileout, seed)
        calc.generate_data_mad(fileout)

if __name__ == "__main__":
    main()
