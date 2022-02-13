# raivBox

raivBox is a standalone neural audio synthesis device based on Google Magenta's DDSP library, with an NVIDIA Jetson Nano 2GB embedded development board at its core. This entire project is designed to be as affordable and replicable as possible, while still offering a robust neural synthesis experience outside of a traditional laptop/desktop computer environment. raivBox is currently under development; this repository is not yet complete.

![raivBox logo](https://github.com/jacktipper/raivBox/blob/main/assets/raivBox_logo.png)

The `raivBox/` folder contains all of the important files and scripts.

To initialize a blank NVIDIA Jetson Nano 2GB operating system img with the required dependencies for this project, first flash the OS onto the Jetson, complete the basic user setup wizard, and then follow the instructions in the header of `raivBox/raivShell.sh`.

In later versions of this repository, a fully-baked OS img file will be provided. Flashing this img to a Jetson Nano 2GB's SD card will pre-load all of the software needed for this project.

The core DDSP synthesis loop is in `raivBox/synth.py`.

Useful system stats and UI animations are displayed via `raivBox/stats.py`.

All of the additional python files in this repository facilitate hardware-software interaction on the physical raivBox device.

The `assets/` folder contains a few graphics, as well as the three .stl files neede to 3D-print the raivBox housing components.

#

### To replicate this project, you will need:

• 1x NVIDIA Jetson Nano 2GB Embedded Development Kit

• 1x micro SD card (>=32GB)

• 1x compatible USB-C power supply (https://www.canakit.com/raspberry-pi-4-power-supply.html)

• 1x USB audio DAC (https://www.sabrent.com/product/AU-EMCB/usb-aluminum-external-stereo-sound-adapter-black/)

• 1x 3.5mm microphone (https://www.bhphotovideo.com/c/product/1632609-REG/boya_by_um4_3_5mm_mini_flexible_lavalier.html)


### For the optional housing and hardware control interface, you will also need:

• Access to a capable 3D-printer or 3D-printing service for the .stl files in the `assets/` folder

• Custom PCB manufacturing for `assets/raivBox_pcb.zip`

• 1x compatible fan with hardware (https://smile.amazon.com/dp/B07YFDCGQV/ref=cm_sw_em_r_mt_dp_BDKKRKYSQ2KYJ76BYARS)

• 2x 2-pin 30mm arcade buttons (https://smile.amazon.com/gp/product/B005BZ421M/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&psc=1)

• 3x 10kΩ potentiometers (https://smile.amazon.com/gp/product/B07B64MWRF/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&psc=1)

• 3x standard 2-pin 5mm LEDs

• 1x MCP-3008 ADC IC chip

• 3x 220Ω resistors

• 2x 1kΩ resistors


![raivBox product](https://github.com/jacktipper/raivBox/blob/main/assets/raivBox_product.jpeg)
