Directories are organized as numbered steps, with different versions indexed within

`00_Blank_IMG` has blank, default OEM installs, without core dependencies

	Images are flashed onto the micro-SD card using the `Raspberry Pi Imager` app.

`01_Init_Script` has the installation scripts needed to load core dependencies on blank installs

	Shell scripts are copied to the Jetson Nano via `scp` (secure copy) using SSH in terminal.
	Commands are used as follows:
	
	% scp raivShell.sh jt@aotunano.local:Desktop/

`02_Backup_IMG` has full OS snapshots of different "personalized" installs, with and without core dependencies

	Backup images are created using the `ApplePi-Baker v2.2.3` app, saving to a `.zip` file
	with `BACKUP: Shrink to Minimum Size` enabled. Backup images can be flashed onto the
	micro-SD card using the `Raspberry Pi Imager` app, just like a clean install.

`03_DDSP_Script` attempts to introduce DDSP to the system



