# aurora-hpc

[![LaTeX build](../../actions/workflows/pdflatex.yml/badge.svg)](../../actions/workflows/pdflatex.yml)
[![Tech-report](https://img.shields.io/badge/PDF-TechReport-orange.svg?style=flat)](../gh-action-result/pdf-techreport/documents/tech-report/tech-report-aurora-dawn.pdf)

Exploration of running Aurora on HPC.

## Abstract

Dawn is one of the two AI-optimised High Performance Computing systems funded under the AI Research Resource programme in the UK.
Unlike other similar systems it uses Intel® Data Center GPU Max 1550 GPUs, which are incompatible with the *defacto* standard NVIDIA CUDA and AMD ROCm accelerator frameworks.

This work explores the ease of porting code originally developed for NVIDIA hardware onto the Intel hardware on Dawn.
We used the Microsoft Aurora weather model as a test case, benchmarking inference and training speeds and comparing against the NVIDIA A100 GPUs used on the Baskerville High Performance Computing system.

We discover and discuss some practical and unexpected differences, including some of the challenges of redeploying to a new framework, configurations differences that have a significant impact on performance and differences in the actual results output.
