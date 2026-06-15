# AGENTS.md

## Project Overview

This repository contains an implementation of CS336 Assignment 1: Basics.

The project uses Python and is managed with `uv`. Source code lives under
`cs336_basics/`, and the assignment test suite lives under `tests/`.

## Development Principles

Implementations should prioritize correctness and clarity before performance
optimization. Keep changes focused on the assignment component being worked on,
and avoid unrelated refactors.

The public interfaces expected by the assignment tests are defined in
`tests/adapters.py`. Implementation code should be connected through those
adapters rather than changing test expectations.
