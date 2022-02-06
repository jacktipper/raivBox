#!/bin/bash
# Run this script to properly initialize and run 'synth.py'.
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
source ~/.bashrc
echo 'Initialized' ; echo
python3 synth.py