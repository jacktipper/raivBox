# raivBox

Directories are organized as numbered steps, with different versions indexed within

`Init_Script` has the installation scripts needed to load core dependencies on blank installs

>Shell scripts are copied to the Jetson Nano via `scp` (secure copy) using SSH in terminal. Commands are used as follows:

	% scp raivShell.sh jt@aotunano.local:Desktop/


`DDSP_Script` introduces DDSP to the system


`Analog_Ctrl` adds hardware control interfacing with the underlying software