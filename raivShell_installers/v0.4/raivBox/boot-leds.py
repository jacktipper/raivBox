from time import sleep
import Jetson.GPIO as GPIO
import os
import subprocess
os.system('./init-py.sh')

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

directory = os.getcwd()
GO = None
booting = True

while booting:
    b1 = GPIO.input(button1pin)
    b2 = GPIO.input(button2pin)
    if b1 == 0 and b2 == 0:
        GPIO.output(led0pin, GPIO.LOW)
        GPIO.output(led1pin, GPIO.LOW)
        GPIO.output(led2pin, GPIO.LOW)
        booting = False
        GPIO.cleanup()
        GO = subprocess.Popen(['python3 buttons.py'],
                              cwd=directory, shell=True)
    sleep(0.02)
