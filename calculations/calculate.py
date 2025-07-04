#!/usr/bin/env python
# vim: et:ts=4:sts=4:sw=4

# SPDX-License-Identifier: MIT
# Copyright 2025 The Alan Turing Institute

import torch
import struct

device="mps"

# See https://docs.pytorch.org/xla/master/tutorials/precision_tutorial.html
def binary_fraction_to_fp32(bstr: str) -> float:
  if bstr[:4] != "0b1.":
    raise ValueError(f"Invalid binary string: {bstr}")
  fraction_bits = bstr[4:]
  mantissa = 1.0
  for i, bit in enumerate(fraction_bits):
    mantissa += int(bit) * 2**-(i + 1)
  return float(mantissa)


def fp32_to_binary_fraction(fp32_float: float) -> str:
  x_bytes = struct.pack(">f", fp32_float)  # Big-endian IEEE 754 float32
  as_int = struct.unpack(">I", x_bytes)[0]  # Interpret bits as uint32
  sign = (as_int >> 31) & 0b1
  exponent = (as_int >> 23) & 0xFF
  mantissa = as_int & 0x7FFFFF  # lower 23 bits
  return f"FORMAT:0b SIGN:{sign} EXPONENT:{exponent:08b} MANTISSA:{mantissa:023b} VALUE={fp32_float}"


def get_rand_matrix():
  """Returns a diagonal matrix of shape 1024, 1024, values between 0.999 and 1.111"""
  eye = torch.eye(1024, dtype=torch.float32, device=device)
  rand_ = torch.rand(
      (1024, 1024), dtype=torch.float32, device=device) * 0.2 + 0.9
  result = eye * rand_
  assert torch.nonzero(result).size(0) == 1024, torch.nonzero(result).size(0)
  return result

binary = "0b1.0001"
result = binary_fraction_to_fp32(binary)
print("Binary: {}".format(binary))
print("Result: {}".format(result))

binary = fp32_to_binary_fraction(result)
print("Binary: {}".format(binary))

matrix_a = get_rand_matrix()
matrix_b = get_rand_matrix()
matrix_c = matrix_a.matmul(matrix_b)
print("a[0, 0] = {}".format(matrix_a[0][0]))
print("b[0, 0] = {}".format(matrix_b[0][0]))
print("c[0, 0] = {}".format(matrix_c[0][0]))
result = matrix_a[0][0] * matrix_b[0][0]
print("a[0, 0] * b[0, 0] = {}".format(result))
