import os
from time import sleep
import busio
import digitalio
import board
import adafruit_mcp3xxx.mcp3008 as MCP
from adafruit_mcp3xxx.analog_in import AnalogIn

# create the spi bus
spi = busio.SPI(clock=board.SCK, MISO=board.MISO, MOSI=board.MOSI)

# create the chip selection (cs)
cs = digitalio.DigitalInOut(board.D5)

# create the mcp object
mcp = MCP.MCP3008(spi, cs)

# create the analog input channels
c0 = AnalogIn(mcp, MCP.P0)
c2 = AnalogIn(mcp, MCP.P2)
c4 = AnalogIn(mcp, MCP.P4)

print('ADC Values and Voltages:')
print('Channel 0: ' + str(c0.value) + ', ' + str(c0.voltage) + 'V')
print('Channel 2: ' + str(c2.value) + ', ' + str(c2.voltage) + 'V')
print('Channel 4: ' + str(c4.value) + ', ' + str(c4.voltage) + 'V')

in_last_read = 0 # keeps track of last pot values
out_last_read = 0
tolerance = 300 # reduces jitter

# using c4 to control the output volume for now

def remap_range(value, left_min, left_max, right_min, right_max):
    # determines range width for each trim adjustment
    left_span = left_max - left_min
    right_span = right_max - right_min
    
    # cast as integers
    valueScaled = int(value - left_min) / int(left_span)
    return int(right_min + (valueScaled * right_span))

while True:
    # assume the pots didn't move yet
    trim_input_changed = False
    trim_output_changed = False

    # read the analog pins
    trim_input = c4.value
    trim_output = c2.value
    
    # calculate change since previous read
    input_adjust = abs(trim_input - in_last_read)
    output_adjust = abs(trim_output - out_last_read)
    
    if input_adjust > tolerance:
        trim_input_changed = True
    if output_adjust > tolerance:
        trim_output_changed = True
    
    if trim_input_changed:
        # convert 16bit adc value into 0-100 volume level
        set_volume = remap_range(trim_input, 0, 65535, 0, 100)
        
        # set OS volume
        print('Volume = {volume}%' .format(volume = set_volume))
        set_vol_cmd = 'pactl set-source-volume 1 {volume}%' \
        .format(volume = set_volume)
        os.system(set_vol_cmd)
        
        # save the pot reading for the next loop
        in_last_read = trim_input
    
    if trim_output_changed:
        # convert 16bit adc value into 0-100 volume level
        set_volume = remap_range(trim_output, 0, 65535, 0, 100)
        
        # set OS volume
        print('Volume = {volume}%' .format(volume = set_volume))
        set_vol_cmd = 'pactl set-sink-volume 0 {volume}%' \
        .format(volume = set_volume)
        os.system(set_vol_cmd)
        
    # wait for a moment to save processing power
    sleep(0.5)

