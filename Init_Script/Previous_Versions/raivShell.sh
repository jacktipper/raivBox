#!/bin/bash

# RAIVSHELL for NVIDIA Jetson Nano 2GB - Deep Learning for Music - Installation Sequence v0.1
#     assembled by Jack Tipper - tipper@nyu.edu - (c) 2021 RAIV - MIT License

# First, put this script file on the Desktop, and then run the following line in terminal (without the #):
# sudo chmod +x ~/Desktop/raivShell.sh ; sudo ~/Desktop/raivShell.sh

clear ; echo '--Initialize RAIVSHELL Installation Sequence for NVIDIA Jetson Nano 2GB--' ; echo ; sleep 1 ; echo '*'
echo ; sleep 1 ; echo '*' ; echo ; sleep 1 ; echo '*' ; echo ; sleep 1 ; echo '*' ; echo ; sleep 1 ; echo '*'
cd ~ ; clear ; echo '--User Confirmation--' ; echo ; sleep 1 ; echo '    This full installation takes approximately 3.5 hours.' ; echo
read -p "    Enter username (case-sensitive): " USER ; echo ; echo "    Got it, your username is: $USER." ; echo ; sleep 1
read -p "    Enter a VNC password: " PASS ; echo ; echo "    Thanks! Use that password to log in to VNC." ; echo ; sleep 1
echo '    Additional interactive input will be required in about 5 minutes.' ; echo
sleep 1 ; echo '*' ; echo ; sleep 1 ; echo '*' ; echo ; sleep 1 ; clear
echo '--Boosting System to FULL THROTTLE--' ; echo ; echo ; sleep 2
sudo nvpmodel -m 0 ; sudo jetson_clocks ; cd /etc/X11/app-defaults/ ; sudo sed -i 's/0:10:00/6:00:00/' ./XScreenSaver ; cd ~ ; clear
echo '--Updating--' ; echo ; echo ; sleep 2 ; sudo apt update ; sudo apt-mark hold systemd ; clear

echo '--Setting Up VNC Software--' ; echo ; echo ; sleep 2
sudo apt install vino ; mkdir -p ~/.config/autostart ; cp /usr/share/applications/vino-server.desktop ~/.config/autostart
gsettings set org.gnome.Vino prompt-enabled false ; gsettings set org.gnome.Vino require-encryption false
gsettings set org.gnome.Vino authentication-methods "['vnc']" ; gsettings set org.gnome.Vino vnc-password $(echo -n "$PASS"|base64) ; clear

echo '--Correcting VNC Settings--' ; echo ; echo ; sleep 2 ; cd /etc/X11/
echo | sudo tee -a xorg.conf ; echo 'Section "Monitor"' | sudo tee -a xorg.conf ; echo '    Identifier "DSI-0"' | sudo tee -a xorg.conf
echo '    Option    "Ignore"' | sudo tee -a xorg.conf ; echo "EndSection" | sudo tee -a xorg.conf ; echo | sudo tee -a xorg.conf
echo 'Section "Screen"' | sudo tee -a xorg.conf ; echo '    Identifier  "Default Screen"' | sudo tee -a xorg.conf
echo '    Monitor "Configured Monitor"' | sudo tee -a xorg.conf ; echo '    Device      "Default Device"' | sudo tee -a xorg.conf
echo '    SubSection "Display"' | sudo tee -a xorg.conf ; echo "        Depth   24" | sudo tee -a xorg.conf
echo "        Virtual 1920 1080" | sudo tee -a xorg.conf ; echo "    EndSubSection" | sudo tee -a xorg.conf
echo "EndSection" | sudo tee -a xorg.conf ; clear

# echo "--Setting Up Automatic Login for $USER--" ; echo ; echo ; sleep 2
# cd ~/etc/lightdm/lightdm.conf.d/ ; echo "autologin-user=$USER" | sudo tee -a 50-nvidia.conf
# echo "autologin-user-timeout=0" | sudo tee -a 50-nvidia.conf ; cd ~ ; clear

echo '--Making Development Folder and Purging Bloatware--' ; echo ; echo ; sleep 2 ; cd ~/Desktop ; sudo -u $USER mkdir Development ; cd ~
yes | sudo apt-get purge libreoffice* ; sudo apt-get clean ; clear
shut
echo '--Installing Jetson Inference Module from NVIDIA--' ; echo ; sleep 2
echo '    Be prepared to make menu selections at the end of this installation stage.' ; sleep 2
echo '    After selecting your models and PyTorch installation, there will not be'
echo '    any additional interactive elements in this script.' ; echo ; sleep 1 ; echo '*' ; echo ; sleep 1 ; echo '*' ; sleep 1 ; echo
git clone https://github.com/dusty-nv/jetson-inference ; cd jetson-inference ; git submodule update --init
mkdir build ; cd build ; cmake ../ ; clear

echo '--Installing Python Libraries and Dependencies--' ; echo ; echo '    This section will take about 3 hours to complete.'
echo ; sleep 1 ; echo '*' ; echo ; sleep 1 ; echo '*' ; echo ; sleep 1 ; echo '*' ; echo ; sleep 1 ; echo '*' ; sleep 1 ; echo '*'
yes | sudo apt install python3-pip python3-pil 
yes | sudo apt install llvm-7 ; cd /usr/bin ; sudo ln llvm-config-7 llvm-config ; cd ~
yes | sudo apt-get install git cmake puredata libatlas-base-dev gfortran libhdf5-serial-dev hdf5-tools python3-dev 
yes | sudo apt-get install nano mlocate libpython3-dev python3-numpy libfreetype6-dev python3-setuptools
yes | sudo apt-get install protobuf-compiler libprotobuf-dev openssl libssl-dev libcurl4-openssl-dev
yes | sudo apt-get install cython3 libxml2-dev libxslt1-dev curl libssl1.0-dev nodejs-dev node-gyp nodejs npm
sudo npm install -g n ; sudo n stable ; sudo updatedb

sudo -H pip3 install threadpoolctl dominate ; sudo -H pip3 install -U numpy setuptools ; yes | sudo apt-get upgrade
sudo -H pip3 install llvmlite==0.31 numba==0.48

sudo -H pip3 install matplotlib ipython colorgram.py scikit-build
sudo -H pip3 install librosa opencv-python pandas ; clear

echo '--Installing Visual Studio Code (Code-OSS)--' ; echo ; echo ; sleep 2
curl -L https://github.com/toolboc/vscode/releases/download/1.32.3/code-oss_1.32.3-arm64.deb -o code-oss_1.32.3-arm64.deb
sudo dpkg -i code-oss_1.32.3-arm64.deb ; clear

echo '--Installing Jupyter--' ; echo ; echo ; sleep 2
sudo -H pip3 install jupyter jupyterlab
sudo jupyter labextension install @jupyter-widgets/jupyterlab-manager
sudo jupyter labextension install @jupyterlab/statusbar ; clear

# New Additions Sept 2021

echo '--Installing DDSP--' ; echo ; echo ; sleep 2
read -p "    Enter GitHub username (case-sensitive): " GHUSER ; echo ; echo "    Got it, your username is: $GHUSER." ; echo ; sleep 1
read -p "    Enter GitHub token: " GHPASS ; echo ; echo "    Thanks!" ; echo ; sleep 1
git clone https://$GHUSER:$GHPASS@github.com/acids-ircam/ddsp_pytorch ; cd ddsp_pytorch ; cd realtime ; mkdir build ; cd build
export Torch_DIR=~/opt/anaconda3/lib/python3.8/site-packages/torch/share/cmake/Torch
cmake ../ -DCMAKE_PREFIX_PATH=~/opt/anaconda3/lib/python3.8/site-packages/torch -DCMAKE_BUILD_TYPE=Release
make install
echo '--Installing ADC Connections for Analog Control--' ; echo ; echo ; sleep 2
sudo -H pip3 install adafruit-circuitpython-mcp3xxx Adafruit-Blinka ; clear

# End New Additions Sept 2021

echo '--Final Housekeeping--' ; echo ; sleep 2
git clone https://github.com/JetsonHacksNano/resizeSwapMemory ; cd resizeSwapMemory ; ./setSwapMemorySize.sh -g 16
sudo apt update ; yes | sudo apt upgrade ; sudo -H pip3 install -U jetson-stats ; yes | sudo apt-get install florence
yes | sudo apt autoremove ; sudo apt clean ; clear

echo '--SPI Configuration and Reboot Sequence--' ; echo ; sleep 2
echo '    First, please resize this terminal window to be much larger on your screen.' ; sleep 1
echo '    Then, in the next screen, select "Configure 40-pin expansion header"'
echo '    at the bottom of the menu.' ; sleep 1
echo '    After that, make sure [*] spi1 (19,21,23,24,26) is selected.' ; sleep 1
echo '    Then select "Back" at the bottom, save, and reboot.' ; sleep 1
read -p "    When you are ready to proceed to these steps, please press [ENTER] to proceed." x ; sudo /opt/nvidia/jetson-io/jetson-io.py

# End of Shell