#!/bin/bash

#Run the following line in terminal (without the #) to create the systemd modules:
#sudo chmod +x ~/Desktop/raivShell/analogCtrl/analogShell.sh ; sudo ~/Desktop/raivShell/analogCtrl/analogShell.sh

git clone https://github.com/JetsonHacksNano/installPiOLED
cd installPiOLED
./installPiOLED.sh
./createService.sh
sudo scp ~/Desktop/raivShell/analogCtrl/stats.py /usr/local/lib/python3.6/dist-packages/pioled
sudo systemctl enable pioled_stats
sudo systemctl start pioled_stats

cd ~/Desktop/raivShell/analogCtrl ; mkdir Audio

sudo touch /etc/systemd/user/boot-leds.service
sudo cat ~/Desktop/raivShell/analogCtrl/boot-leds_service.txt > /etc/systemd/user/boot-leds.service
sudo chmod +x /etc/systemd/user/boot-leds.service ; 
systemctl --user enable boot-leds
systemctl --user start boot-leds

sudo touch /etc/systemd/system/boot-knobs.service
sudo cat ~/Desktop/raivShell/analogCtrl/boot-knobs_service.txt > /etc/systemd/system/boot-knobs.service
sudo chmod +x /etc/systemd/system/boot-knobs.service ; 
sudo systemctl enable boot-knobs
sudo systemctl start boot-knobs

sleep 1 ; clear