import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
ctrl_pin = 21
GPIO.setup(ctrl_pin, GPIO.OUT, initial=GPIO.LOW)

for i in range(12):
    GPIO.output(ctrl_pin, GPIO.HIGH)
    sleep(0.1)
    GPIO.output(ctrl_pin, GPIO.LOW)
    sleep(0.1)

GPIO.output(ctrl_pin, GPIO.HIGH)
