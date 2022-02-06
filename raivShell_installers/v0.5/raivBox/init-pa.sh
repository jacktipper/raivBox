#!/bin/bash
# Run this script to properly initialie PulseAudio during boot.
pacmd set-default-sink 0
pacmd set-default-source 1