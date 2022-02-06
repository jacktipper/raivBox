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
from time import sleep
import Jetson.GPIO as GPIO
import os
import subprocess
os.system('./init-py.sh')


"""Set up GPIO control/interaction for the LEDs and control buttons."""
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

led0pin = 20
led1pin = 21
led2pin = 16
button1pin = 27
button2pin = 26

GPIO.setup(led0pin, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(led1pin, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(led2pin, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(button1pin, GPIO.IN)
GPIO.setup(button2pin, GPIO.IN)


"""Send a boot-up blink sequence to the three red status LEDs."""
for i in range(3):
    GPIO.output(led0pin, GPIO.HIGH)
    sleep(0.08)
    GPIO.output(led0pin, GPIO.LOW)
    sleep(0.12)
    GPIO.output(led1pin, GPIO.HIGH)
    sleep(0.08)
    GPIO.output(led1pin, GPIO.LOW)
    sleep(0.12)
    GPIO.output(led2pin, GPIO.HIGH)
    sleep(0.08)
    GPIO.output(led2pin, GPIO.LOW)
    sleep(0.12)
for i in range(4):
    GPIO.output(led0pin, GPIO.HIGH)
    sleep(0.04)
    GPIO.output(led0pin, GPIO.LOW)
    sleep(0.06)
    GPIO.output(led1pin, GPIO.HIGH)
    sleep(0.04)
    GPIO.output(led1pin, GPIO.LOW)
    sleep(0.06)
    GPIO.output(led2pin, GPIO.HIGH)
    sleep(0.04)
    GPIO.output(led2pin, GPIO.LOW)
    sleep(0.06)
GPIO.output(led0pin, GPIO.HIGH)
sleep(0.2)
GPIO.output(led0pin, GPIO.LOW)
sleep(0.06)
GPIO.output(led0pin, GPIO.HIGH)
GPIO.output(led1pin, GPIO.HIGH)
GPIO.output(led2pin, GPIO.HIGH)


"""This loop and `try/except` listens for when both of the hardware
buttons are simultaneously depressed, and runs 'buttons.py' once
the condition is met. The `going` variable is used for the subprocess
that runs 'buttons.py', and `gone` is the flag for whether or not
`going` has been executed. 
"""
directory = os.getcwd()
going = None
gone = False
booting = True

try:
    while booting:
        b1 = GPIO.input(button1pin)
        b2 = GPIO.input(button2pin)
        if b1 == 0 and b2 == 0:
            GPIO.output(led0pin, GPIO.LOW)
            GPIO.output(led1pin, GPIO.LOW)
            GPIO.output(led2pin, GPIO.LOW)
            booting = False
            GPIO.cleanup()
            going = subprocess.Popen(['python3 buttons.py'],
                                  cwd=directory, shell=True)
            gone = True
        sleep(0.02)
except:
    """If the while loop returns an exception and 'buttons.py' has 
    not been executed, this `except` statement attempts to rectify
    the issue by calling 'buttons.py' itself.
    """
    if not gone:
        going = subprocess.Popen(['python3 buttons.py'],
                              cwd=directory, shell=True)