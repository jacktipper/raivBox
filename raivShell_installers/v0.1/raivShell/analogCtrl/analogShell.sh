#!/bin/bash

sudo scp ~/Desktop/raivShell/analogCtrl/led.py ~/Desktop/Development/Boot
sudo scp ~/Desktop/raivShell/analogCtrl/monitor-volume.py ~/Desktop/Development/Boot
sudo touch /etc/systemd/system/led.service
sudo cat ~/Desktop/raivShell/analogCtrl/led_service.txt > /etc/systemd/system/led.service
sudo chmod +x /etc/systemd/system/led.service ; sudo systemctl enable led ; sudo systemctl start led
sudo touch /etc/systemd/system/monitor-volume.service
sudo cat ~/Desktop/raivShell/analogCtrl/monitor-volume_service.txt > /etc/systemd/system/monitor-volume.service
sudo chmod +x /etc/systemd/system/monitor-volume.service ; sudo systemctl enable monitor-volume
sudo systemctl start monitor-volume ; sleep 1 ; clear