#! /usr/bin/python3
# Copyright (c) 2017 Adafruit Industries
# Author: Tony DiCola & James DeVito
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# Portions copyright (c) NVIDIA 2019
# Portions copyright (c) JetsonHacks 2019

import time

import Adafruit_SSD1306   # This is the driver chip for the Adafruit PiOLED

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import subprocess


def get_network_interface_state(interface):
    return subprocess.check_output('cat /sys/class/net/%s/operstate' % interface, shell=True).decode('ascii')[:-1]


def get_ip_address(interface):
    if get_network_interface_state(interface) == 'down':
        return None
    cmd = "ifconfig %s | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1'" % interface
    return subprocess.check_output(cmd, shell=True).decode('ascii')[:-1]

# Return a string representing the percentage of CPU in use


def get_cpu_usage():
    # Shell scripts for system monitoring from here : https://unix.stackexchange.com/questions/119126/command-to-display-memory-usage-disk-usage-and-cpu-load
    cmd = "top -bn1 | grep load | awk '{printf \"CPU Load: %.2f\", $(NF-2)}'"
    CPU = subprocess.check_output(cmd, shell=True).decode('ascii')[:-1]
    return CPU

# Return a float representing the percentage of GPU in use.
# On the Jetson Nano, the GPU is GPU0


def get_gpu_usage():
    GPU = 0.0
    with open("/sys/devices/gpu.0/load", encoding="utf-8") as gpu_file:
        GPU = gpu_file.readline()
        GPU = int(GPU)/10
    return GPU


# 128x32 display with hardware I2C:
# setting gpio to 1 is hack to avoid platform detection
disp = Adafruit_SSD1306.SSD1306_128_32(rst=None, i2c_bus=1, gpio=1)

# Initialize library.
disp.begin()

# Clear display.
disp.clear()
disp.display()

# Create blank image for drawing.
# Make sure to create image with mode '1' for 1-bit color.
width = disp.width
height = disp.height
image = Image.new('1', (width, height))

# Get drawing object to draw on image.
draw = ImageDraw.Draw(image)

# Draw a black filled box to clear the image.
draw.rectangle((0, 0, width, height), outline=0, fill=0)

# Draw some shapes.
# First define some constants to allow easy resizing of shapes.
padding = -2
top = padding
bottom = height-padding
# Move left to right keeping track of the current x position for drawing shapes.
x = 0

# Load default font.
font = ImageFont.load_default()

booting = True
while booting:           # "1234567890123456789012"
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "x                     ", font=font, fill=255)
    draw.text((x, top+8),  "                      ", font=font, fill=255)
    draw.text((x, top+16), "                      ", font=font, fill=255)
    draw.text((x, top+25), "                      ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "ox                    ", font=font, fill=255)
    draw.text((x, top+8),  "x                     ", font=font, fill=255)
    draw.text((x, top+16), "                      ", font=font, fill=255)
    draw.text((x, top+25), "                      ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "Box                   ", font=font, fill=255)
    draw.text((x, top+8),  "ox                    ", font=font, fill=255)
    draw.text((x, top+16), "x                     ", font=font, fill=255)
    draw.text((x, top+25), "                      ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "vBox                  ", font=font, fill=255)
    draw.text((x, top+8),  "Box                   ", font=font, fill=255)
    draw.text((x, top+16), "ox                    ", font=font, fill=255)
    draw.text((x, top+25), "x                     ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "ivBox                 ", font=font, fill=255)
    draw.text((x, top+8),  "vBox                  ", font=font, fill=255)
    draw.text((x, top+16), "Box                   ", font=font, fill=255)
    draw.text((x, top+25), "ox                    ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "aivBox                ", font=font, fill=255)
    draw.text((x, top+8),  "ivBox                 ", font=font, fill=255)
    draw.text((x, top+16), "vBox                  ", font=font, fill=255)
    draw.text((x, top+25), "Box                   ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "raivBox               ", font=font, fill=255)
    draw.text((x, top+8),  "aivBox                ", font=font, fill=255)
    draw.text((x, top+16), "ivBox                 ", font=font, fill=255)
    draw.text((x, top+25), "vBox                  ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    " raivBox              ", font=font, fill=255)
    draw.text((x, top+8),  "raivBox               ", font=font, fill=255)
    draw.text((x, top+16), "aivBox                ", font=font, fill=255)
    draw.text((x, top+25), "ivBox                 ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "  raivBox             ", font=font, fill=255)
    draw.text((x, top+8),  " raivBox              ", font=font, fill=255)
    draw.text((x, top+16), "raivBox               ", font=font, fill=255)
    draw.text((x, top+25), "aivBox                ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "   raivBox            ", font=font, fill=255)
    draw.text((x, top+8),  "  raivBox             ", font=font, fill=255)
    draw.text((x, top+16), " raivBox              ", font=font, fill=255)
    draw.text((x, top+25), "raivBox               ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "    raivBox           ", font=font, fill=255)
    draw.text((x, top+8),  "   raivBox            ", font=font, fill=255)
    draw.text((x, top+16), "  raivBox             ", font=font, fill=255)
    draw.text((x, top+25), " raivBox              ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "     raivBox          ", font=font, fill=255)
    draw.text((x, top+8),  "    raivBox           ", font=font, fill=255)
    draw.text((x, top+16), "   raivBox            ", font=font, fill=255)
    draw.text((x, top+25), "  raivBox             ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "      raivBox         ", font=font, fill=255)
    draw.text((x, top+8),  "     raivBox          ", font=font, fill=255)
    draw.text((x, top+16), "    raivBox           ", font=font, fill=255)
    draw.text((x, top+25), "   raivBox            ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "       raivBox        ", font=font, fill=255)
    draw.text((x, top+8),  "      raivBox         ", font=font, fill=255)
    draw.text((x, top+16), "     raivBox          ", font=font, fill=255)
    draw.text((x, top+25), "    raivBox           ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "        raivBox       ", font=font, fill=255)
    draw.text((x, top+8),  "       raivBox        ", font=font, fill=255)
    draw.text((x, top+16), "      raivBox         ", font=font, fill=255)
    draw.text((x, top+25), "     raivBox          ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "         raivBox      ", font=font, fill=255)
    draw.text((x, top+8),  "        raivBox       ", font=font, fill=255)
    draw.text((x, top+16), "       raivBox        ", font=font, fill=255)
    draw.text((x, top+25), "      raivBox         ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "          raivBox     ", font=font, fill=255)
    draw.text((x, top+8),  "         raivBox      ", font=font, fill=255)
    draw.text((x, top+16), "        raivBox       ", font=font, fill=255)
    draw.text((x, top+25), "       raivBox        ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "           raivBox    ", font=font, fill=255)
    draw.text((x, top+8),  "          raivBox     ", font=font, fill=255)
    draw.text((x, top+16), "         raivBox      ", font=font, fill=255)
    draw.text((x, top+25), "        raivBox       ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "            raivBox   ", font=font, fill=255)
    draw.text((x, top+8),  "           raivBox    ", font=font, fill=255)
    draw.text((x, top+16), "          raivBox     ", font=font, fill=255)
    draw.text((x, top+25), "         raivBox      ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "             raivBox  ", font=font, fill=255)
    draw.text((x, top+8),  "            raivBox   ", font=font, fill=255)
    draw.text((x, top+16), "           raivBox    ", font=font, fill=255)
    draw.text((x, top+25), "          raivBox     ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "              raivBox ", font=font, fill=255)
    draw.text((x, top+8),  "             raivBox  ", font=font, fill=255)
    draw.text((x, top+16), "            raivBox   ", font=font, fill=255)
    draw.text((x, top+25), "           raivBox    ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "               raivBox", font=font, fill=255)
    draw.text((x, top+8),  "              raivBox ", font=font, fill=255)
    draw.text((x, top+16), "             raivBox  ", font=font, fill=255)
    draw.text((x, top+25), "            raivBox   ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "                raivBo", font=font, fill=255)
    draw.text((x, top+8),  "               raivBox", font=font, fill=255)
    draw.text((x, top+16), "              raivBox ", font=font, fill=255)
    draw.text((x, top+25), "             raivBox  ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "                 raivB", font=font, fill=255)
    draw.text((x, top+8),  "                raivBo", font=font, fill=255)
    draw.text((x, top+16), "               raivBox", font=font, fill=255)
    draw.text((x, top+25), "              raivBox ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "                  raiv", font=font, fill=255)
    draw.text((x, top+8),  "                 raivB", font=font, fill=255)
    draw.text((x, top+16), "                raivBo", font=font, fill=255)
    draw.text((x, top+25), "               raivBox", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "                   rai", font=font, fill=255)
    draw.text((x, top+8),  "                  raiv", font=font, fill=255)
    draw.text((x, top+16), "                 raivB", font=font, fill=255)
    draw.text((x, top+25), "                raivBo", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "                    ra", font=font, fill=255)
    draw.text((x, top+8),  "                   rai", font=font, fill=255)
    draw.text((x, top+16), "                  raiv", font=font, fill=255)
    draw.text((x, top+25), "                 raivB", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "                     r", font=font, fill=255)
    draw.text((x, top+8),  "                    ra", font=font, fill=255)
    draw.text((x, top+16), "                   rai", font=font, fill=255)
    draw.text((x, top+25), "                  raiv", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "                      ", font=font, fill=255)
    draw.text((x, top+8),  "                     r", font=font, fill=255)
    draw.text((x, top+16), "                    ra", font=font, fill=255)
    draw.text((x, top+25), "                   rai", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "                      ", font=font, fill=255)
    draw.text((x, top+8),  "                      ", font=font, fill=255)
    draw.text((x, top+16), "                     r", font=font, fill=255)
    draw.text((x, top+25), "                    ra", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "                      ", font=font, fill=255)
    draw.text((x, top+8),  "                      ", font=font, fill=255)
    draw.text((x, top+16), "                      ", font=font, fill=255)
    draw.text((x, top+25), "                     r", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "     R A I V B O X    ", font=font, fill=255)
    draw.text((x, top+8),  "                      ", font=font, fill=255)
    draw.text((x, top+16), "                      ", font=font, fill=255)
    draw.text((x, top+25), "                      ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.24)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "                      ", font=font, fill=255)
    draw.text((x, top+8),  "    R A I V B O X     ", font=font, fill=255)
    draw.text((x, top+16), "                      ", font=font, fill=255)
    draw.text((x, top+25), "                      ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.24)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "                      ", font=font, fill=255)
    draw.text((x, top+8),  "                      ", font=font, fill=255)
    draw.text((x, top+16), "     R A I V B O X    ", font=font, fill=255)
    draw.text((x, top+25), "                      ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.24)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "                      ", font=font, fill=255)
    draw.text((x, top+8),  "                      ", font=font, fill=255)
    draw.text((x, top+16), "                      ", font=font, fill=255)
    draw.text((x, top+25), "    R A I V B O X     ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.24)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "                      ", font=font, fill=255)
    draw.text((x, top+8),  "          by          ", font=font, fill=255)
    draw.text((x, top+16), "      Jack Tipper     ", font=font, fill=255)
    draw.text((x, top+25), "                      ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(1.14)
    booting = False


while True:

    # Draw a black filled box to clear the image.
    draw.rectangle((0, 0, width, height), outline=0, fill=0)

    # Shell scripts for system monitoring from here : https://unix.stackexchange.com/questions/119126/command-to-display-memory-usage-disk-usage-and-cpu-load
    cmd = "free -m | awk 'NR==2{printf \" Mem:  %.0f%% %s/%sM \", $3*100/$2, $3,$2 }'"
    MemUsage = subprocess.check_output(cmd, shell=True)
    cmd = "df -h | awk '$NF==\"/\"{printf \" Disk: %s %d/%dGB \", $5, $3,$2 }'"
    Disk = subprocess.check_output(cmd, shell=True)

    # Show the current neural synthesizer model instead of GPU
    Model = subprocess.check_output("cat ~/Desktop/raivBox/models/model.txt", shell=True).decode('ascii')[:-1]
    draw.text((x, top), str(" Model: " + Model.upper()), font=font, fill=255)

    # Print the IP address
    # Two examples here, wired and wireless
    draw.text((x, top+8), " IP:   " + str(get_ip_address('wlan0')),  font=font, fill=255)

    # Alternate solution: Draw the GPU usage as text
    # draw.text((x, top+8),     "GPU:  " +"{:3.1f}".format(GPU)+" %", font=font, fill=255)
    # We draw the GPU usage as a bar graph
    string_width, string_height = font.getsize(" GPU:   ")
    # Figure out the width of the bar
    full_bar_width = width-(x+string_width)-1
    gpu_usage = get_gpu_usage()
    # Avoid divide by zero ...
    # if gpu_usage == 0.0:
    #     gpu_usage = 0.001
    # draw_bar_width = int(full_bar_width*(gpu_usage/100))
    # draw.text((x, top+8),     "GPU:  ", font=font, fill=255)
    # draw.rectangle((x+string_width, top+12, x+string_width +
    #                 draw_bar_width, top+14), outline=1, fill=1)

    # Show the memory Usage
    draw.text((x, top+16), str(MemUsage.decode('utf-8')), font=font, fill=255)
    
    draw.text((x, top+25), str(Disk.decode('utf-8')), font=font, fill=255)
    # cmd = "pactl list sources | grep '^[[:space:]]Volume:' | head -n $(( $SOURCE + 2 )) | tail -n 1 | grep -oP '[0-9]+[0-9]+(?=%)'"
    # InVol = subprocess.check_output(cmd, shell=True).decode('ascii')[:-1]
    # cmd = "pactl list sinks | grep '^[[:space:]]Volume:' | head -n $(( $SINK + 1 )) | tail -n 1 | grep -oP '[0-9]+[0-9]+(?=%)'"
    # OutVol = subprocess.check_output(cmd, shell=True).decode('ascii')[:-1]
    
    # if InVol > 5 or OutVol > 5:
    #     # Show the amount of disk being used
    #     draw.text((x, top+25), str(Disk.decode('utf-8')), font=font, fill=255)
    # else:
    #     draw.text((x, top+25), str(" IN LVL: " + InVol + "  OUT LVL: " + OutVol), font=font, fill=255)

    # Display image.
    # Set the SSD1306 image to the PIL image we have made, then dispaly
    disp.image(image)
    disp.display()
    # 1.0 = 1 second; The divisor is the desired updates (frames) per second
    time.sleep(0.25)
