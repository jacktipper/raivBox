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
"""Import libraries, and also initialize python and pulseaudio."""
from time import sleep
import Jetson.GPIO as GPIO
import signal
import subprocess
import os
os.system('./init-py.sh')
os.system('./init-pa.sh')


"""Set up GPIO control for the bottom status LED."""
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
led0pin = 20
GPIO.setup(led0pin, GPIO.OUT, initial=GPIO.LOW)


"""Run a blink sequence on the LED to signify 'buttons.py' load-in."""
for i in range(3):
    GPIO.output(led0pin, GPIO.HIGH)
    sleep(0.08)
    GPIO.output(led0pin, GPIO.LOW)
    sleep(0.12)
for i in range(4):
    GPIO.output(led0pin, GPIO.HIGH)
    sleep(0.04)
    GPIO.output(led0pin, GPIO.LOW)
    sleep(0.06)
sleep(0.26)
GPIO.output(led0pin, GPIO.HIGH)


"""Set up the additional LEDs and buttons for GPIO."""
led1pin = 21
led2pin = 16
button1pin = 27
button2pin = 26

GPIO.setup(led1pin, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(button1pin, GPIO.IN)
GPIO.setup(led2pin, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(button2pin, GPIO.IN)


"""Initialize variables and commands."""
prev1value = None
prev2value = None
recording = 0
playing = 0
rec = None
synth = None
play = None

directory = os.getcwd()
rec_cmd = str(
    "exec arecord -d 20 -t wav -c 1 -f FLOAT_LE -r 16000 audio/input.wav")


"""This `if/else` statement checks if the synthesizer in 'synth.py' present. 
If it has been moved or renamed by the user, this will instruct the raivBox
to just record and play audio directly instead of erroring out.
"""
if os.path.exists('synth.py'):
    play1cmd = str("exec aplay audio/output.wav")
    # Initialize the synth loop
    synth = subprocess.Popen(  
        ["./init-synth.sh"], cwd=directory, shell=True)
else:
    play1cmd = str("exec aplay audio/input.wav")
play2cmd = str("exec aplay audio/input.wav")


"""Call the core processing loop."""
try:
    while True:
        curr1value = GPIO.input(button1pin)
        curr2value = GPIO.input(button2pin)
        if curr1value != prev1value:
            GPIO.output(led1pin, not curr1value)
            # 'curr1value == 0' is the on-press condition
            if curr1value == 0:
                recording = 1
                print('Recording In Progress')
                rec = subprocess.Popen(
                    [rec_cmd], cwd=directory, stdout=subprocess.PIPE, shell=True)
                sleep(0.15)
            # Synthesize output audio once recording is complete
            elif recording == 1 and curr1value == 1:
                rec.send_signal(signal.SIGINT)
                recording = 0
                print('Recording Ended')
                os.system('touch flags/read.y') # Flag that the input is fully recorded
            prev1value = curr1value
        if curr2value != prev2value:
            GPIO.output(led2pin, not curr2value)
            if curr2value == 0:
                playing = 1
                print('Playing Recorded Audio')
                if os.path.exists('audio/output.wav'):
                    play = subprocess.Popen(
                        [play1cmd], cwd=directory, stdout=subprocess.PIPE, shell=True)
                elif os.path.exists('audio/input.wav'):
                    play = subprocess.Popen(
                        [play2cmd], cwd=directory, stdout=subprocess.PIPE, shell=True)
                else:
                    pass
            if playing == 1 and curr2value == 1:
                try:
                    play.send_signal(signal.SIGINT)
                except:
                    pass
                playing = 0
                print('Audio Stopped')
            prev2value = curr2value
        # Set the status check interval for the while loop (responsivity)
        sleep(0.02)
except:
    """Note: This `except` doesn't work as intended yet."""
    # If the program is prematurely terminated, turn on just the two top LEDs to indicate
    GPIO.output(led0pin, GPIO.LOW)
    GPIO.output(led1pin, GPIO.HIGH)
    GPIO.output(led2pin, GPIO.HIGH)
    # As a precaution, attempt to interrupt the synth if an exception is thrown
    try:
        synth.send_signal(signal.SIGINT)
    except:
        pass
    # After 5 seconds, execute 'boot-leds.py' again to completely restart the synthesizer
    sleep(5)
    os.system('python3 boot-leds.py')