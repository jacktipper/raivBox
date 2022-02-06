from adafruit_mcp3xxx.analog_in import AnalogIn
import adafruit_mcp3xxx.mcp3008 as MCP
import board
import digitalio
import busio
from time import sleep
import os
os.system('./init-py.sh')
os.system('./init-pa.sh')

# create the spi bus
spi = busio.SPI(clock=board.SCK, MISO=board.MISO, MOSI=board.MOSI)

# create the chip selection (cs)
cs = digitalio.DigitalInOut(board.D17)

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

in_last_read = 0  # keeps track of last pot values
out_last_read = 0
top_last_read = 0
tolerance = 1000  # reduces jitter


def remap_range(value, left_min, left_max, right_min, right_max):
    # determines range width for each trim adjustment
    left_span = left_max - left_min
    right_span = right_max - right_min

    # cast as integers
    valueScaled = int(value - left_min) / int(left_span)
    return int(right_min + (valueScaled * right_span))


sd_cond_1 = False
sd_cond_2 = False
sd_cond_3 = False
sd_cmd = False


def shutdown_check(value, src):
    if src == 'vol':
        if value < 5:
            condition = True
        else:
            condition = False
    if src == 'mod':
        if value < 5:
            condition = True
        else:
            condition = False
    return condition


def shutdown_cmd(sd_cond_1, sd_cond_2, sd_cond_3):
    if sd_cond_1 == True and sd_cond_2 == True and sd_cond_3 == True:
        sd_cmd = True
    else:
        sd_cmd = False
    return sd_cmd


# initialize input and output volume reports
os.system("echo '01' | tee flags/invol.txt")
os.system("echo '01' | tee flags/outvol.txt")


powered_on = True
model_bank = ['acid', 
              'saxophone',
              'violin',
              'trumpet',
              'flute']
prev_model = model_bank[0] # initialize

while powered_on:
    # assume the pots didn't move yet
    trim_input_changed = False
    trim_output_changed = False
    trim_top_changed = False

    # read the analog pins
    trim_input = c2.value
    trim_output = c4.value
    trim_top = c0.value

    # calculate change since previous read
    input_adjust = abs(trim_input - in_last_read)
    output_adjust = abs(trim_output - out_last_read)
    top_adjust = abs(trim_top - top_last_read)

    if input_adjust > tolerance:
        trim_input_changed = True
    if output_adjust > tolerance:
        trim_output_changed = True
    if top_adjust > tolerance:
        trim_top_changed = True

    if trim_input_changed:
        # convert 16bit adc value into 0-100 volume level
        set_in_vol = remap_range(trim_input, 0, 65535, 0, 100)

        # set OS input volume
        print('Input Volume = {}%' .format(set_in_vol))
        set_in_vol_cmd = 'pactl set-source-volume 1 {}%' \
            .format(set_in_vol)
        os.system(set_in_vol_cmd)
        os.system("echo '{}' | tee flags/invol.txt".format(int(set_in_vol)))

        # save the pot reading for the next loop
        in_last_read = trim_input

        sd_cond_1 = shutdown_check(set_in_vol, 'vol')

    if trim_output_changed:
        # convert 16bit adc value into 0-100 volume level
        set_out_vol = remap_range(trim_output, 0, 65535, 0, 100)

        # set OS output volume
        print('Output Volume = {}%' .format(set_out_vol))
        set_out_vol_cmd = 'pactl set-sink-volume 0 {}%' \
            .format(set_out_vol)
        os.system(set_out_vol_cmd)
        os.system("echo '{}' | tee flags/outvol.txt".format(int(set_out_vol)))

        # save the pot reading for the next loop
        out_last_read = trim_output

        sd_cond_2 = shutdown_check(set_out_vol, 'vol')

    if trim_top_changed:
        # convert 16bit adc value into 0-255 level
        set_model = remap_range(trim_top, 0, 65535, 0, 100)

        # set the neural synthesizer model
        if set_model < 2:
            # model = 'power down'
            model = model_bank[0]
        elif set_model <= 20:
            model = model_bank[0]
        elif set_model <= 40:
            model = model_bank[1]
        elif set_model <= 60:
            model = model_bank[2]
        elif set_model <= 80:
            model = model_bank[3]
        else:
            model = model_bank[4]
        if model != prev_model:
            os.system("echo '{}' | sudo tee flags/model.txt" .format(model))
        prev_model = model

        # save the pot reading for the next loop
        top_last_read = trim_top

        sd_cond_3 = shutdown_check(set_model, 'mod')

    # wait for a moment to save processing power
    sleep(0.02)

    sd_cmd = shutdown_cmd(sd_cond_1, sd_cond_2, sd_cond_3)
    if sd_cmd == True:
        powered_on = False
        os.system("echo '0' | sudo tee flags/state.txt")
        os.system("echo '255' | sudo tee /sys/devices/pwm-fan/target_pwm")
        os.system('python3 sd-leds.py')
        sleep(0.34)
        os.system("echo '1' | sudo tee flags/state.txt")
        os.system('sudo shutdown now')