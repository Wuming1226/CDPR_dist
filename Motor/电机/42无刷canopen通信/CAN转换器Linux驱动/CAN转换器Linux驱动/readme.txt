1、执行以下命令在linux下安装libusb1.0
   sudo apt-get install libusb
   sudo apt-get install libusb-1.0
2、执行以下命令拷贝动态库到/usr/lib
   sudo cp libNiMotionUSBCAN.so /usr/lib
3、允许普通用户访问USB设备
   (1) 执行以下命令添加usbfs用户组
       sudo groupadd usbfs
   (2) 执行以下命令将用户添加到usbfs用户组
       sudo usermod -a -G usbfs qihuanming
   (3) 将61-NiMotionUSBCAN.rules文件拷贝到/etc/udev/rules.d目录
       sudo cp 61-NiMotionUSBCAN.rules /etc/udev/rules.d