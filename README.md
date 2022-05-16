# raivBox

The raivBox is a standalone neural audio synthesis device based on Google Magenta's [DDSP](https://github.com/magenta/ddsp), with an [NVIDIA Jetson Nano 2GB](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-nano/education-projects/) embedded development board at its core. This entire project is designed to be as affordable and replicable as possible, while still offering a robust neural synthesis experience outside of a traditional laptop/desktop computer environment.

![raivBox logo](https://github.com/jacktipper/raivBox/blob/main/assets/raivBox_logo.png)

## Abstract

Recent advances in deep learning-based audio creation are fueling the rise of a new approach to sound design: neural synthesis. Until now, robust neural synthesizers for musical sounds have been relegated to traditional desktop and cloud-based computing environments. Hosting these generative systems on low-cost devices will broaden offline and untethered access to this fledgling technology, helping to make way for a new era of modular music and audio experimentation. In this thesis, the implementation of neural waveform synthesis on a resource-limited embedded development platform is explored using a modified version of Google Magenta’s Differentiable Digital Signal Processing (DDSP) timbre transfer pipeline on an Nvidia Jetson Nano 2GB embedded prototyping board. Timbre transfer is a type of neural synthesis achieved by extracting audio features from an input signal and utilizing those data to generate a new waveform in the target timbre. This project introduces the raivBox: a functional neural synthesizer in the form of an affordable, standalone hardware product. It features a physical control interface that provides responsive audio input and output, a selection of five neural timbre models to choose from, and a straightforward user experience. The reworked neural synthesis pipeline demonstrates a significant processing latency reduction when compared with its predecessor, while maintaining similar sonic outputs. Qualitative research data indicate that users find the raivBox to be both intuitive and fun to interact with, and also suggest that it is perceived to be a valuable new tool for music creation and sound design. As this project is designed to be readily replicable by end users, the raivBox source code, build instructions, print files, and a variety of audio examples are all publicly available via its GitHub repository.

## Getting Started

Please refer to `assets/raivBox_thesis.pdf` for the full documentation of this project. See the *Methodology* section for detailed build instructions.

The `raivBox/` folder contains all of the important files and scripts.

To initialize a blank NVIDIA Jetson Nano 2GB operating system img with the required dependencies for this project, first flash the OS onto the Jetson, complete the basic user setup wizard, and then follow the instructions in the header of `raivBox/raivShell.sh`.

In later versions of this repository, a fully-baked OS img file will be provided. Flashing this img to a Jetson Nano 2GB's SD card will pre-load all of the software needed for this project.

The core DDSP synthesis loop is in `raivBox/synth.py`.

Useful system stats and UI animations are displayed via `raivBox/stats.py`.

All of the additional python files in this repository facilitate hardware-software interaction on the physical raivBox device.

In addition to the master's thesis document, the `assets/` folder contains the custom printed circuit board (PCB) zip file and the three stl files needed to 3D-print the raivBox housing components, as well as a few pictures and renders.

#

![raivBox software architecture](https://github.com/jacktipper/raivBox/blob/main/assets/raivBox_arch.png)

#

### To replicate the basic functionality of this project, you will need:

• 1x NVIDIA Jetson Nano 2GB Embedded Development Kit

• 1x micro SD card (32GB+)

• 1x compatible USB-C power supply (https://www.canakit.com/raspberry-pi-4-power-supply.html)

• 1x Sabrent USB audio adapter (https://www.sabrent.com/product/AU-EMCB/usb-aluminum-external-stereo-sound-adapter-black/)

• 1x 3.5mm microphone (https://www.bhphotovideo.com/c/product/1632609-REG/boya_by_um4_3_5mm_mini_flexible_lavalier.html)


### For the optional housing and hardware control interface, you will also need:

• Access to a capable 3D-printer or 3D-printing service for the .stl files in the `assets/` folder

• Custom PCB manufacturing for `assets/raivBox_pcb.zip`

• 1x Adafruit PiOLED display (https://www.adafruit.com/product/3527)

• 1x compatible fan with hardware (https://smile.amazon.com/dp/B07YFDCGQV/ref=cm_sw_em_r_mt_dp_BDKKRKYSQ2KYJ76BYARS)

• 2x 2-pin 30mm arcade buttons (https://smile.amazon.com/gp/product/B005BZ421M/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&psc=1)

• 3x 10kΩ potentiometers (https://smile.amazon.com/gp/product/B07B64MWRF/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&psc=1)

• 3x standard 2-pin 5mm LEDs

• 1x MCP-3008 ADC IC chip

• 3x 220Ω resistors

• 2x 1kΩ resistors

• Plenty of M2.5 standoffs (https://smile.amazon.com/gp/product/B075K3QBMX/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&psc=1)

• Plenty of insulated wire (~28 gauge)

• Basic soldering tools and know-how

• 40-pin stacking header kit for PCB attachment

• 1x INIU USB-C power bank (https://smile.amazon.com/gp/product/B07CZDXDG8/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&psc=1)

• 1x short USB-C male to USB-C male cable

#

![raivBox product](https://github.com/jacktipper/raivBox/blob/main/assets/raivBox_product.jpeg)

<!-- ![raivBox render](https://github.com/jacktipper/raivBox/blob/main/assets/raivBox_render.png) -->
