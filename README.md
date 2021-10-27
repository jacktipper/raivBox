# raivBox

raivBox is an accessible standalone neural synthesizer based on Google Magenta's DDSP library, with an NVIDIA Jetson Nano 2GB embedded prototyping system at its core. This entire project is designed to be as affordable and replicable as possible, while still offering a robust neural synthesis experience outside of a traditional laptop/desktop computer environment.

#### Files:

`Init_Script` has the installation scripts needed to load core dependencies on blank installs

>Shell scripts are copied to the Jetson Nano via `scp` (secure copy) using SSH in terminal. Commands are used as follows:

	% scp raivShell.sh jt@aotunano.local:Desktop/


`DDSP_Script` introduces DDSP to the system


`Analog_Ctrl` adds hardware control interfacing with the underlying software