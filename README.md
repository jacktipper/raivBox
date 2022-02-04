# raivBox

raivBox is a standalone digital neural audio synthesis device based on Google Magenta's DDSP library, with an NVIDIA Jetson Nano 2GB embedded prototyping system at its core. This entire project is designed to be as affordable and replicable as possible, while still offering a robust neural synthesis experience outside of a traditional laptop/desktop computer environment. raivBox is currently under development; this repository is not yet complete.

![raivBox Logo](https://github.com/jacktipper/raivBox/blob/main/Assets/raivBox_logo.png)

#### In the first version:

`Init_Script` has the installation scripts needed to load core dependencies on blank installs

`DDSP_Script` introduces DDSP to the system

`Analog_Ctrl` adds hardware control interfacing with the underlying software

#### In the upcoming version:

`raivShell` has the installation scripts needed for loading the core dependencies on blank installs

`raivSynth` is the DDSP synthesizer module

`raivCtrl` is the hardware control module
