from time import sleep
import Jetson.GPIO as GPIO
import signal
import subprocess
import os
os.system('./init-py.sh') # initialize for python
os.system('./init-pa.sh') # initialize pulseaudio

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
# set the 'ON' indicator LED to led0 and map it to the correct BCM GPIO number
led0pin = 20
GPIO.setup(led0pin, GPIO.OUT, initial=GPIO.LOW)

# blink led0 during the initialization sequence to indicate success
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

# leave led0 on
GPIO.output(led0pin, GPIO.HIGH)

# map the additional LEDs and buttons to the correct GPIO pins
led1pin = 21
button1pin = 27
led2pin = 16
button2pin = 26
# initialize the loop values to 'None' or '0'
prev1value = None
prev2value = None
recording = 0
playing = 0
rec = None
synth = None
play = None

GPIO.setup(led1pin, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(button1pin, GPIO.IN)
GPIO.setup(led2pin, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(button2pin, GPIO.IN)

directory = os.getcwd()
rec_cmd = str(
    "exec arecord -d 20 -t wav -c 1 -f FLOAT_LE -r 16000 audio/input.wav")

if os.path.exists('synth.sh'):
    play_cmd = str("exec aplay audio/output.wav")
    os.system('./init-synth.sh') # initialize the synth loop
else:
    play_cmd = str("exec aplay audio/input.wav")

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
            # synthesize output audio once recording is complete
            elif recording == 1 and curr1value == 1:
                rec.send_signal(signal.SIGINT)
                recording = 0
                print('Recording Ended')
            prev1value = curr1value
        if curr2value != prev2value:
            GPIO.output(led2pin, not curr2value)
            if curr2value == 0:
                playing = 1
                print('Playing Recorded Audio')
                try:
                    play = subprocess.Popen(
                        [play1cmd], cwd=directory, stdout=subprocess.PIPE, shell=True)
                except:
                    play = subprocess.Popen(
                        [play2cmd], cwd=directory, stdout=subprocess.PIPE, shell=True)
            if playing == 1 and curr2value == 1:
                play.send_signal(signal.SIGINT)
                playing = 0
                print('Audio Stopped')
            prev2value = curr2value
        # set the status check interval for the while loop (responsivity)
        sleep(0.02)
finally:
    # if the program is prematurely terminated, turn on just the two top LEDs to indicate
    GPIO.output(led0pin, GPIO.LOW)
    GPIO.output(led1pin, GPIO.HIGH)
    GPIO.output(led2pin, GPIO.HIGH)
    sleep(5)
    os.system('python3 boot-leds.py')