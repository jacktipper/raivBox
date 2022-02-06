#! /usr/bin/python3
# Portions copyright (c) 2022 RAIV
# Author: Jack Tipper
#
# Portions copyright (c) 2017 Adafruit Industries
# Authors: Tony DiCola & James DeVito
#
# Portions copyright (c) JetsonHacks 2019
#
# Portions copyright (c) NVIDIA 2019
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

# Lint as: python3
"""Libraries and drivers."""
import time
import subprocess
import Adafruit_SSD1306  # This is the driver chip for the Adafruit PiOLED

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


"""These functions assist in gathering data to display on the PiOLED."""
def get_network_interface_state(interface):
    return subprocess.check_output('cat /sys/class/net/%s/operstate' % interface, shell=True).decode('ascii')[:-1]


def get_ip_address(interface):
    if get_network_interface_state(interface) == 'down':
        return None
    cmd = "ifconfig %s | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1'" % interface
    return subprocess.check_output(cmd, shell=True).decode('ascii')[:-1]


def get_cpu_usage():
    # Shell scripts for system monitoring from here: 
    # https://unix.stackexchange.com/questions/119126/command-to-display-memory-usage-disk-usage-and-cpu-load
    cmd = "awk '{u=$2+$4; t=$2+$4+$5; if (NR==1){u1=u; t1=t;} else print ($2+$4-u1) * 100 / (t-t1) \"%\"; }' \
           <(grep 'cpu ' /proc/stat) <(sleep 0.16;grep 'cpu ' /proc/stat)"
    CPU = float(subprocess.check_output(cmd, shell=True).decode('ascii')[:-3])
    return CPU


def get_gpu_usage():
    GPU = 0.0
    with open("/sys/devices/gpu.0/load", encoding="utf-8") as gpu_file:
        GPU = gpu_file.readline()
        GPU = int(GPU)/10
    return GPU


"""The display is 128x32 pixels, with hardware I2C.
Setting gpio to 1 is hack to avoid platform detection.
"""
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


"""During the initial boot-up sequence of the raivBox, display an ASCII animation."""
booting = True
while booting:
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "                      ", font=font, fill=255)
    draw.text((x, top+8),  "     R A I V B O X    ", font=font, fill=255)
    draw.text((x, top+16), "                      ", font=font, fill=255)
    draw.text((x, top+25), "                      ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(1.14)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    disp.image(image)
    disp.display()
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "x                     ", font=font, fill=255)
    draw.text((x, top+8),  "                      ", font=font, fill=255)
    draw.text((x, top+16), "                      ", font=font, fill=255)
    draw.text((x, top+25), "                      ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.04)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "ox                    ", font=font, fill=255)
    draw.text((x, top+8),  "x                     ", font=font, fill=255)
    draw.text((x, top+16), "                      ", font=font, fill=255)
    draw.text((x, top+25), "                      ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.04)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "Box                   ", font=font, fill=255)
    draw.text((x, top+8),  "ox                    ", font=font, fill=255)
    draw.text((x, top+16), "x                     ", font=font, fill=255)
    draw.text((x, top+25), "                      ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.04)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "vBox                  ", font=font, fill=255)
    draw.text((x, top+8),  "Box                   ", font=font, fill=255)
    draw.text((x, top+16), "ox                    ", font=font, fill=255)
    draw.text((x, top+25), "x                     ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.04)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "ivBox                 ", font=font, fill=255)
    draw.text((x, top+8),  "vBox                  ", font=font, fill=255)
    draw.text((x, top+16), "Box                   ", font=font, fill=255)
    draw.text((x, top+25), "ox                    ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.04)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "aivBox                ", font=font, fill=255)
    draw.text((x, top+8),  "ivBox                 ", font=font, fill=255)
    draw.text((x, top+16), "vBox                  ", font=font, fill=255)
    draw.text((x, top+25), "Box                   ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.04)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "raivBox               ", font=font, fill=255)
    draw.text((x, top+8),  "aivBox                ", font=font, fill=255)
    draw.text((x, top+16), "ivBox                 ", font=font, fill=255)
    draw.text((x, top+25), "vBox                  ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.04)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    " raivBox              ", font=font, fill=255)
    draw.text((x, top+8),  "raivBox               ", font=font, fill=255)
    draw.text((x, top+16), "aivBox                ", font=font, fill=255)
    draw.text((x, top+25), "ivBox                 ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.04)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "  raivBox             ", font=font, fill=255)
    draw.text((x, top+8),  " raivBox              ", font=font, fill=255)
    draw.text((x, top+16), "raivBox               ", font=font, fill=255)
    draw.text((x, top+25), "aivBox                ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.04)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "   raivBox            ", font=font, fill=255)
    draw.text((x, top+8),  "  raivBox             ", font=font, fill=255)
    draw.text((x, top+16), " raivBox              ", font=font, fill=255)
    draw.text((x, top+25), "raivBox               ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.04)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "    raivBox           ", font=font, fill=255)
    draw.text((x, top+8),  "   raivBox            ", font=font, fill=255)
    draw.text((x, top+16), "  raivBox             ", font=font, fill=255)
    draw.text((x, top+25), " raivBox              ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.04)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "     raivBox          ", font=font, fill=255)
    draw.text((x, top+8),  "    raivBox           ", font=font, fill=255)
    draw.text((x, top+16), "   raivBox            ", font=font, fill=255)
    draw.text((x, top+25), "  raivBox             ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.04)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "      raivBox         ", font=font, fill=255)
    draw.text((x, top+8),  "     raivBox          ", font=font, fill=255)
    draw.text((x, top+16), "    raivBox           ", font=font, fill=255)
    draw.text((x, top+25), "   raivBox            ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.04)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "       raivBox        ", font=font, fill=255)
    draw.text((x, top+8),  "      raivBox         ", font=font, fill=255)
    draw.text((x, top+16), "     raivBox          ", font=font, fill=255)
    draw.text((x, top+25), "    raivBox           ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.04)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "        raivBox       ", font=font, fill=255)
    draw.text((x, top+8),  "       raivBox        ", font=font, fill=255)
    draw.text((x, top+16), "      raivBox         ", font=font, fill=255)
    draw.text((x, top+25), "     raivBox          ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.04)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "         raivBox      ", font=font, fill=255)
    draw.text((x, top+8),  "        raivBox       ", font=font, fill=255)
    draw.text((x, top+16), "       raivBox        ", font=font, fill=255)
    draw.text((x, top+25), "      raivBox         ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.04)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "          raivBox     ", font=font, fill=255)
    draw.text((x, top+8),  "         raivBox      ", font=font, fill=255)
    draw.text((x, top+16), "        raivBox       ", font=font, fill=255)
    draw.text((x, top+25), "       raivBox        ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.04)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "           raivBox    ", font=font, fill=255)
    draw.text((x, top+8),  "          raivBox     ", font=font, fill=255)
    draw.text((x, top+16), "         raivBox      ", font=font, fill=255)
    draw.text((x, top+25), "        raivBox       ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.04)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "            raivBox   ", font=font, fill=255)
    draw.text((x, top+8),  "           raivBox    ", font=font, fill=255)
    draw.text((x, top+16), "          raivBox     ", font=font, fill=255)
    draw.text((x, top+25), "         raivBox      ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.04)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "             raivBox  ", font=font, fill=255)
    draw.text((x, top+8),  "            raivBox   ", font=font, fill=255)
    draw.text((x, top+16), "           raivBox    ", font=font, fill=255)
    draw.text((x, top+25), "          raivBox     ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.04)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "              raivBox ", font=font, fill=255)
    draw.text((x, top+8),  "             raivBox  ", font=font, fill=255)
    draw.text((x, top+16), "            raivBox   ", font=font, fill=255)
    draw.text((x, top+25), "           raivBox    ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.04)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "               raivBox", font=font, fill=255)
    draw.text((x, top+8),  "              raivBox ", font=font, fill=255)
    draw.text((x, top+16), "             raivBox  ", font=font, fill=255)
    draw.text((x, top+25), "            raivBox   ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.04)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "                raivBo", font=font, fill=255)
    draw.text((x, top+8),  "               raivBox", font=font, fill=255)
    draw.text((x, top+16), "              raivBox ", font=font, fill=255)
    draw.text((x, top+25), "             raivBox  ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.04)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "                 raivB", font=font, fill=255)
    draw.text((x, top+8),  "                raivBo", font=font, fill=255)
    draw.text((x, top+16), "               raivBox", font=font, fill=255)
    draw.text((x, top+25), "              raivBox ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.04)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "                  raiv", font=font, fill=255)
    draw.text((x, top+8),  "                 raivB", font=font, fill=255)
    draw.text((x, top+16), "                raivBo", font=font, fill=255)
    draw.text((x, top+25), "               raivBox", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.04)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "                   rai", font=font, fill=255)
    draw.text((x, top+8),  "                  raiv", font=font, fill=255)
    draw.text((x, top+16), "                 raivB", font=font, fill=255)
    draw.text((x, top+25), "                raivBo", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.04)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "                    ra", font=font, fill=255)
    draw.text((x, top+8),  "                   rai", font=font, fill=255)
    draw.text((x, top+16), "                  raiv", font=font, fill=255)
    draw.text((x, top+25), "                 raivB", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.04)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "                     r", font=font, fill=255)
    draw.text((x, top+8),  "                    ra", font=font, fill=255)
    draw.text((x, top+16), "                   rai", font=font, fill=255)
    draw.text((x, top+25), "                  raiv", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.04)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "                      ", font=font, fill=255)
    draw.text((x, top+8),  "                     r", font=font, fill=255)
    draw.text((x, top+16), "                    ra", font=font, fill=255)
    draw.text((x, top+25), "                   rai", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.04)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "                      ", font=font, fill=255)
    draw.text((x, top+8),  "                      ", font=font, fill=255)
    draw.text((x, top+16), "                     r", font=font, fill=255)
    draw.text((x, top+25), "                    ra", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.04)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "                      ", font=font, fill=255)
    draw.text((x, top+8),  "                      ", font=font, fill=255)
    draw.text((x, top+16), "                      ", font=font, fill=255)
    draw.text((x, top+25), "                     r", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.04)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
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
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "     R A I V B O X    ", font=font, fill=255)
    draw.text((x, top+8),  "     R A I V B O X    ", font=font, fill=255)
    draw.text((x, top+16), "                      ", font=font, fill=255)
    draw.text((x, top+25), "                      ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "     R A I V B O X    ", font=font, fill=255)
    draw.text((x, top+8),  "     R A I V B O X    ", font=font, fill=255)
    draw.text((x, top+16), "     R A I V B O X    ", font=font, fill=255)
    draw.text((x, top+25), "                      ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "     R A I V B O X    ", font=font, fill=255)
    draw.text((x, top+8),  "     R A I V B O X    ", font=font, fill=255)
    draw.text((x, top+16), "     R A I V B O X    ", font=font, fill=255)
    draw.text((x, top+25), "     R A I V B O X    ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "                      ", font=font, fill=255)
    draw.text((x, top+8),  "     R A I V B O X    ", font=font, fill=255)
    draw.text((x, top+16), "     R A I V B O X    ", font=font, fill=255)
    draw.text((x, top+25), "     R A I V B O X    ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "                      ", font=font, fill=255)
    draw.text((x, top+8),  "                      ", font=font, fill=255)
    draw.text((x, top+16), "     R A I V B O X    ", font=font, fill=255)
    draw.text((x, top+25), "     R A I V B O X    ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "                      ", font=font, fill=255)
    draw.text((x, top+8),  "                      ", font=font, fill=255)
    draw.text((x, top+16), "                      ", font=font, fill=255)
    draw.text((x, top+25), "     R A I V B O X    ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    disp.image(image)
    disp.display()
    time.sleep(0.06)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((x, top),    "                      ", font=font, fill=255)
    draw.text((x, top+8),  "          by          ", font=font, fill=255)
    draw.text((x, top+16), "      Jack Tipper     ", font=font, fill=255)
    draw.text((x, top+25), "                      ", font=font, fill=255)
    disp.image(image)
    disp.display()
    time.sleep(1.14)
    booting = False


"""Now that the raivBox is powered up, start the core refresh loop and display system stats."""
powered_on = True
while powered_on:

    # Draw a black filled box to clear the image.
    draw.rectangle((0, 0, width, height), outline=0, fill=0)

    # Shell scripts for system monitoring from here: 
    # https://unix.stackexchange.com/questions/119126/command-to-display-memory-usage-disk-usage-and-cpu-load
    cmd = "free -m | awk 'NR==2{printf \" Mem:  %.0f%% %s/%sM \", $3*100/$2, $3,$2 }'"
    MemUsage = subprocess.check_output(cmd, shell=True)
    cmd = "df -h | awk '$NF==\"/\"{printf \" Disk: %s %d/%dGB \", $5, $3,$2 }'"
    Disk = subprocess.check_output(cmd, shell=True)

    # Show the current neural synthesizer model instead of GPU
    Model = subprocess.check_output("cat ~/Desktop/raivBox/flags/model.txt", shell=True).decode('ascii')[:-1]
    draw.text((x, top), str(" Model: " + Model.upper()), font=font, fill=255)

    # Print the IP address
    # Two examples here, wired and wireless
    if str(get_ip_address('wlan0')) is not 'None':
        draw.text((x, top+8), " IP:   " + str(get_ip_address('wlan0')),  font=font, fill=255)
    else:
        if True:
            # Draw the GPU usage as a bar graph
            string_width, string_height = font.getsize(" GPU:  ")
            # Figure out the width of the bar
            full_bar_width = width-(x+string_width)-1
            gpu_usage = get_gpu_usage()
            # Avoid divide by zero
            if gpu_usage == 0.0:
                gpu_usage = 0.001
            draw_bar_width = int(full_bar_width*(gpu_usage/100))
            draw.text((x, top+8), " GPU:  ", font=font, fill=255)
            draw.rectangle((x+string_width, top+12, x+string_width +
                            draw_bar_width, top+14), outline=1, fill=1)
        else:
            # Draw the CPU usage as a bar graph
            string_width, string_height = font.getsize(" CPU:  ")
            # Figure out the width of the bar
            full_bar_width = width-(x+string_width)-1
            cpu_usage = get_cpu_usage()
            # Avoid divide by zero
            if cpu_usage == 0.0:
                cpu_usage = 0.001
            draw_bar_width = int(full_bar_width*(cpu_usage/100))
            draw.text((x, top+8), " CPU:  ", font=font, fill=255)
            draw.rectangle((x+string_width, top+12, x+string_width +
                            draw_bar_width, top+14), outline=1, fill=1)

    # Show the memory usage.
    draw.text((x, top+16), str(MemUsage.decode('utf-8')), font=font, fill=255)


    """Extract the input and output volume settings from their respective flag files."""
    try:
        InVol = int(subprocess.check_output("cat ~/Desktop/raivBox/flags/invol.txt", shell=True).decode('ascii'))
    except:
        InVol = 0
    try:
        OutVol = int(subprocess.check_output("cat ~/Desktop/raivBox/flags/outvol.txt", shell=True).decode('ascii'))
    except:
        OutVol = 0
    if InVol > 99: InVol = '99'
    if OutVol > 99: OutVol = '99'
    if InVol < 10: InVol = ' {}'.format(InVol)
    if OutVol < 10: OutVol = ' {}'.format(OutVol)


    """When either volume level is below 50%, display both levels on the PiOLED screen.
    This allows the user to see when recording/output levels are too low, and thus assists 
    with hardware debugging.
    """
    if int(InVol) <= 50 or int(OutVol) <= 50:
        # Show the volume levels.
        draw.text((x, top+25), " IN: " + str(InVol) + "%    OUT: " + str(OutVol) + "%", font=font, fill=255)   
    else:
        # Show the amount of disk being used.
        draw.text((x, top+25), str(Disk.decode('utf-8')), font=font, fill=255)


    """Display image by setting the SSD1306 image to the PIL image we have made, then calling display."""
    disp.image(image)
    disp.display()


    """Check if the state flag has changed to '0', which would indicate that the user has 
    requested that the raivBox shut down. A shut down sequence is initiated when all three
    hardware knobs are set to their lowest setting, as read by 'raivBox/boot-knobs.py'.
    """
    try:
        powered_on = bool(int(subprocess.check_output("cat ~/Desktop/raivBox/flags/state.txt", shell=True).decode('ascii')))
    except:
        powered_on = True


    """When a shut down request is detected, display a brief animation."""
    if not powered_on:
        while not powered_on:
                draw.rectangle((0, 0, width, height), outline=0, fill=0)
                draw.text((x, top),    "            *         ", font=font, fill=255)
                draw.text((x, top+8),  "       SHUTTING       ", font=font, fill=255)
                draw.text((x, top+16), "         DOWN         ", font=font, fill=255)
                draw.text((x, top+25), "         *            ", font=font, fill=255)
                disp.image(image)
                disp.display()
                time.sleep(0.16)
                draw.rectangle((0, 0, width, height), outline=0, fill=0)
                draw.text((x, top),    "           *          ", font=font, fill=255)
                draw.text((x, top+8),  "       SHUTTING       ", font=font, fill=255)
                draw.text((x, top+16), "         DOWN         ", font=font, fill=255)
                draw.text((x, top+25), "          *           ", font=font, fill=255)
                disp.image(image)
                disp.display()
                time.sleep(0.16)
                draw.rectangle((0, 0, width, height), outline=0, fill=0)
                draw.text((x, top),    "          *           ", font=font, fill=255)
                draw.text((x, top+8),  "       SHUTTING       ", font=font, fill=255)
                draw.text((x, top+16), "         DOWN         ", font=font, fill=255)
                draw.text((x, top+25), "           *          ", font=font, fill=255)
                disp.image(image)
                disp.display()
                time.sleep(0.16)
                draw.rectangle((0, 0, width, height), outline=0, fill=0)
                draw.text((x, top),    "         *            ", font=font, fill=255)
                draw.text((x, top+8),  "       SHUTTING       ", font=font, fill=255)
                draw.text((x, top+16), "         DOWN         ", font=font, fill=255)
                draw.text((x, top+25), "            *         ", font=font, fill=255)
                disp.image(image)
                disp.display()
                time.sleep(0.16)
                draw.rectangle((0, 0, width, height), outline=0, fill=0)
                draw.text((x, top),    "          *           ", font=font, fill=255)
                draw.text((x, top+8),  "       SHUTTING       ", font=font, fill=255)
                draw.text((x, top+16), "         DOWN         ", font=font, fill=255)
                draw.text((x, top+25), "           *          ", font=font, fill=255)
                disp.image(image)
                disp.display()
                time.sleep(0.16)
                draw.rectangle((0, 0, width, height), outline=0, fill=0)
                draw.text((x, top),    "           *          ", font=font, fill=255)
                draw.text((x, top+8),  "       SHUTTING       ", font=font, fill=255)
                draw.text((x, top+16), "         DOWN         ", font=font, fill=255)
                draw.text((x, top+25), "          *           ", font=font, fill=255)
                disp.image(image)
                disp.display()
                time.sleep(0.16)
                draw.rectangle((0, 0, width, height), outline=0, fill=0)
                draw.text((x, top),    "                      ", font=font, fill=255)
                draw.text((x, top+8),  "       SHUTTING       ", font=font, fill=255)
                draw.text((x, top+16), "         DOWN         ", font=font, fill=255)
                draw.text((x, top+25), "                      ", font=font, fill=255)
                disp.image(image)
                disp.display()
                time.sleep(0.36)
                draw.rectangle((0, 0, width, height), outline=0, fill=0)
                draw.text((x, top),    "                      ", font=font, fill=255)
                draw.text((x, top+8),  "        ('_')_/       ", font=font, fill=255)
                draw.text((x, top+16), "                      ", font=font, fill=255)
                draw.text((x, top+25), "                      ", font=font, fill=255)
                disp.image(image)
                disp.display()
                time.sleep(0.24)
                draw.rectangle((0, 0, width, height), outline=0, fill=0)
                draw.text((x, top),    "                      ", font=font, fill=255)
                draw.text((x, top+8),  "        ('_')_|       ", font=font, fill=255)
                draw.text((x, top+16), "                      ", font=font, fill=255)
                draw.text((x, top+25), "                      ", font=font, fill=255)
                disp.image(image)
                disp.display()
                time.sleep(0.16)
                draw.rectangle((0, 0, width, height), outline=0, fill=0)
                draw.text((x, top),    "                      ", font=font, fill=255)
                draw.text((x, top+8),  "        ('_')_\       ", font=font, fill=255)
                draw.text((x, top+16), "                      ", font=font, fill=255)
                draw.text((x, top+25), "                      ", font=font, fill=255)
                disp.image(image)
                disp.display()
                time.sleep(0.24)
                draw.rectangle((0, 0, width, height), outline=0, fill=0)
                draw.text((x, top),    "                      ", font=font, fill=255)
                draw.text((x, top+8),  "        ('_')_|       ", font=font, fill=255)
                draw.text((x, top+16), "                      ", font=font, fill=255)
                draw.text((x, top+25), "                      ", font=font, fill=255)
                disp.image(image)
                disp.display()
                time.sleep(0.16)
                draw.rectangle((0, 0, width, height), outline=0, fill=0)
                draw.text((x, top),    "                      ", font=font, fill=255)
                draw.text((x, top+8),  "        (^_^)_/       ", font=font, fill=255)
                draw.text((x, top+16), "                      ", font=font, fill=255)
                draw.text((x, top+25), "                      ", font=font, fill=255)
                disp.image(image)
                disp.display()
                time.sleep(9)
    

    else:
        """The base refresh rate of the stats display is set below (not for animations)."""
        time.sleep(0.16)
