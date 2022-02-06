# Copyright (c) 2022 RAIV - Jack Tipper
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Import libraries and fix the python environment."""
import os
import subprocess
from time import sleep
import Jetson.GPIO as GPIO
os.system('./init-py.sh')


"""Set up the system for GPIO."""
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

led0pin = 20
led1pin = 21
button1pin = 27
led2pin = 16
button2pin = 26

GPIO.setup(led0pin, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(led1pin, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(button1pin, GPIO.IN)
GPIO.setup(led2pin, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(button2pin, GPIO.IN)


"""Run the counterclockwise shut down blink sequence."""
for i in range(4):
    GPIO.output(led0pin, GPIO.HIGH)
    sleep(0.04)
    GPIO.output(led0pin, GPIO.LOW)
    sleep(0.06)
    GPIO.output(led2pin, GPIO.HIGH)
    sleep(0.04)
    GPIO.output(led2pin, GPIO.LOW)
    sleep(0.06)
    GPIO.output(led1pin, GPIO.HIGH)
    sleep(0.04)
    GPIO.output(led1pin, GPIO.LOW)
    sleep(0.06)
for i in range(3):
    GPIO.output(led0pin, GPIO.HIGH)
    sleep(0.08)
    GPIO.output(led0pin, GPIO.LOW)
    sleep(0.12)
    GPIO.output(led2pin, GPIO.HIGH)
    sleep(0.08)
    GPIO.output(led2pin, GPIO.LOW)
    sleep(0.12)
    GPIO.output(led1pin, GPIO.HIGH)
    sleep(0.08)
    GPIO.output(led1pin, GPIO.LOW)
    sleep(0.12)
GPIO.output(led0pin, GPIO.HIGH)
sleep(0.2)
GPIO.output(led0pin, GPIO.LOW)


"""Cleanup the GPIO pins before exiting."""
GPIO.cleanup()