# raivBox

raivBox is a standalone neural audio synthesis device based on Google Magenta's DDSP library, with an NVIDIA Jetson Nano 2GB embedded development board at its core. This entire project is designed to be as affordable and replicable as possible, while still offering a robust neural synthesis experience outside of a traditional laptop/desktop computer environment. raivBox is currently under development; this repository is not yet complete.

![raivBox logo](https://github.com/jacktipper/raivBox/blob/main/graphics/raivBox_logo.png)

The `raivBox/` folder contains all of the important files and scripts.

To initialize a blank NVIDIA Jetson Nano 2GB operating system img with the required dependencies for this project, first flash the OS onto the Jetson, complete the basic user setup wizard, and then follow the instructions in the header of `raivBox/raivShell.sh`.

In later versions of this repository, a hyperlink to a fully-baked OS img will be provided. Flashing this img to a Jetson's SD card will pre-load all of the software needed for this project.

The DDSP synthesis is performed in `raivBox/synth.py`.

Useful system stats and UI animations are displayed via `raivBox/stats.py`.

All of the additional python files in this repository facilitate hardware-software interaction on the physical raivBox device.

![raivBox product](https://github.com/jacktipper/raivBox/blob/main/graphics/raivBox_product.jpeg)
