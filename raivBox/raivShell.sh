#!/bin/bash
# Copyright (c) 2022 RAIV. All Rights Reserved.
#     assembled by Jack Tipper - tipper@nyu.edu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Instructions:
# First, put the enclosing folder on the Desktop, and then run the 
# following line in terminal (without the # at the beginning):
# sudo chmod +x ~/Desktop/raivShell/raivShell.sh ; sudo ~/Desktop/raivShell/raivShell.sh

clear ; echo '--Initialize RAIVSHELL Installation Sequence for NVIDIA Jetson Nano 2GB--' ; echo ; sleep 1 ; echo '*'
echo ; sleep 1 ; echo '*' ; echo ; sleep 1 ; echo '*' ; echo ; sleep 1 ; echo '*' ; echo ; sleep 1 ; echo '*'
cd ~ ; clear ; echo '--User Confirmation--' ; echo ; sleep 1 ; echo '    This full installation takes approximately 3.5 hours.' ; echo
read -p "    Enter username (case-sensitive): " USER ; echo ; echo "    Got it, your username is: $USER." ; echo ; sleep 1
read -p "    Enter a VNC password: " PASS ; echo ; echo "    Thanks! Use that password to log in to VNC." ; echo ; sleep 1
echo '    Additional interactive input will be required in about 5 minutes.' ; echo
sleep 1 ; echo '*' ; echo ; sleep 1 ; echo '*' ; echo ; sleep 1 ; clear
echo '--Boosting System to FULL THROTTLE--' ; echo ; echo ; sleep 2
sudo nvpmodel -m 0 ; sudo jetson_clocks ; cd /etc/X11/app-defaults/ ; sudo sed -i 's/0:10:00/6:00:00/' ./XScreenSaver ; cd ~
git clone https://github.com/JetsonHacksNano/resizeSwapMemory ; cd resizeSwapMemory ; ./setSwapMemorySize.sh -g 16 ; clear
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
echo '--Updating--' ; echo ; echo ; sleep 2 ; sudo apt update ; sudo apt-mark hold systemd ; clear

echo '--Setting Up VNC Software--' ; echo ; echo ; sleep 2
sudo apt install vino ; mkdir -p ~/.config/autostart ; cp /usr/share/applications/vino-server.desktop ~/.config/autostart
gsettings set org.gnome.Vino prompt-enabled false ; gsettings set org.gnome.Vino require-encryption false
gsettings set org.gnome.Vino authentication-methods "['vnc']"
gsettings set org.gnome.Vino vnc-password $(echo -n "$PASS"|base64) ; clear

echo '--Adjusting VNC Settings--' ; echo ; echo ; sleep 2 ; cd /etc/X11/
echo | sudo tee -a xorg.conf ; echo 'Section "Monitor"' | sudo tee -a xorg.confecho '    Identifier "DSI-0"' | sudo tee -a xorg.conf
echo '    Option    "Ignore"' | sudo tee -a xorg.conf ; echo "EndSection" | sudo tee -a xorg.conf ; echo | sudo tee -a xorg.conf
echo 'Section "Screen"' | sudo tee -a xorg.conf ; echo '    Identifier  "Default Screen"' | sudo tee -a xorg.conf
echo '    Monitor "Configured Monitor"' | sudo tee -a xorg.conf ; echo '    Device      "Default Device"' | sudo tee -a xorg.conf
echo '    SubSection "Display"' | sudo tee -a xorg.conf ; echo "        Depth   24" | sudo tee -a xorg.conf
echo "        Virtual 1280 720" | sudo tee -a xorg.conf ; echo "    EndSubSection" | sudo tee -a xorg.conf
echo "EndSection" | sudo tee -a xorg.conf ; clear

echo '--Purging Bloatware--' ; echo ; echo ; sleep 2 ; cd ~
yes | sudo apt-get purge libreoffice* ; sudo apt-get clean ; clear

echo '--Installing Jetson Inference Module from NVIDIA--' ; echo ; sleep 2
echo '    Be prepared to make menu selections at the end of this installation stage.' ; sleep 2
echo '    After selecting your models and installations, there will not be any'
echo '    additional interactive elements in this script until the final reboot sequence.' ; echo ; sleep 1 
echo '*' ; echo ; sleep 1 ; echo '*' ; sleep 1 ; echo
git clone https://github.com/dusty-nv/jetson-inference ; cd jetson-inference ; git submodule update --init
mkdir build ; cd build ; cmake ../ ; yes | sudo dkpg --configure -a ; clear #gets docker set up properly


echo '--Installing Python Libraries and Dependencies--' ; echo ; echo '    The following section takes about 3 hours to complete.'
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
sudo -H pip3 install matplotlib ipython colorgram.py scikit-build ; sudo -H pip3 install librosa pandas ; clear

echo '--Installing DDSP--' ; echo ; echo ; sleep 2
sudo apt-get update ; sudo apt-get install libblas-dev liblapack-dev liblapack-doc
sudo -H pip3 install -U pip testresources setuptools==49.6.0 protobuf
sudo -H pip3 install -U numpy==1.19.4 mock==3.0.5 h5py==2.10.0 keras_preprocessing==1.1.1 # h5py takes a while
sudo -H pip3 install -U keras_applications==1.0.8 gast==0.2.2 futures pybind11
sudo -H pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow
sudo -H npm install -g @bazel/bazelisk # needed for `dm-tree` in DDSP
sudo -H pip3 install -U ddsp colorama ; clear

echo '--Installing ADC Connections for Analog Control--' ; echo ; echo ; sleep 2
sudo -H pip3 install adafruit-circuitpython-mcp3xxx Adafruit-Blinka
cd ../ ; cd raivBox ; sudo chmod +x init-spi.sh ; sudo chmod +x init-pa.sh ; sudo chmod +x init-py.sh ; clear

echo '--Creating OLED Status Display Service--' ; echo ; echo ; sleep 2
sudo touch /etc/systemd/system/pioled-stats.service
sudo cat ~/Desktop/raivBox/pioled-stats_service.txt > /etc/systemd/system/pioled-stats.service
sudo chmod +x /etc/systemd/system/pioled-stats.service
sudo systemctl enable pioled-stats ; sudo systemctl start pioled-stats.service ; clear

echo '--Creating Analog Control systemd Modules--' ; echo ; echo ; sleep 2
sudo touch /etc/systemd/user/boot-leds.service
sudo cat ~/Desktop/raivBox/boot-leds_service.txt > /etc/systemd/user/boot-leds.service
sudo chmod +x /etc/systemd/user/boot-leds.service
systemctl --user enable boot-leds ; systemctl --user start boot-leds ; clear
sudo touch /etc/systemd/system/boot-knobs.service
sudo cat ~/Desktop/raivBox/boot-knobs_service.txt > /etc/systemd/system/boot-knobs.service
sudo chmod +x /etc/systemd/system/boot-knobs.service
sudo systemctl enable boot-knobs ; sudo systemctl start boot-knobs ; clear

echo '--Final Housekeeping--' ; echo ; sleep 2
sudo apt update ; yes | sudo apt upgrade ; sudo -H pip3 install -U jetson-stats ; yes | sudo apt-get install florence
cd ~/Desktop ; touch florence.sh ; echo "#!/bin/bash" | sudo tee -a florence.sh
echo "echo 'Launching Florence Virtual Keyboard'" | sudo tee -a florence.sh
echo "florence" | sudo tee -a florence.sh; sudo chmod +x florence.sh
yes | sudo apt autoremove ; sudo apt clean ; clear

echo '--SPI Configuration and Reboot Sequence--' ; echo ; sleep 2
echo '    First, please resize this terminal window to be much larger on your screen.' ; sleep 1
echo '    Then, in the next screen, select "Configure 40-pin expansion header"'
echo '    at the bottom of the menu.' ; sleep 1
echo '    After that, make sure [*] spi1 (19,21,23,24,26) is selected.' ; sleep 1
echo '    Then select "Back" at the bottom, save, and reboot.' ; sleep 1
read -p "    When you are ready to proceed to these steps, please press [ENTER]." x ; sudo /opt/nvidia/jetson-io/jetson-io.py