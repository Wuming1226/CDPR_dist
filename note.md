### Motor

电机手册及上位机软件（windows）等。

电机 canID 的查看和修改需要使用上位机软件。



### Packages

功能包。

默认电机数量为4，可以在主、从机 cdpr.py 文件中修改电机数量。

#### master

主机代码，部署在笔记本上。

#####  scripts 

- **cdpr.py**：CDPR 基础参数和基本动作，如绳索预紧和放松、设置电机速度、获取末端执行器位置等。主函数现在只调用了预紧函数，可以运行节点实现绳索预紧。
- **Jacobian.py**：计算逆运动学雅各比矩阵。
- **transform.py**：四元数与欧拉角转换，SDO 报文与整形数据转换。
- **track_J.py**：使用滑模控制器的，基于逆运动学雅各比矩阵的控制代码示例。



#### slave

从机代码，部署在书梅派上（除修改电机ID外，基本不需要修改）。

##### srv

原先使用的通讯结构中使用的 service，现已弃用。

##### scripts

- **cdpr.py**：接收主机指令，控制电机运行，回传电机状态。
- **motor.py**：电机控制（SDO 报文指令），主要使用模式（速度模式）设置和速度设置，只被 cdpr.py 调用。
- **transform.py**：同主机。



###  **运行**

#### 主从机通讯设置

为使主从机之间使用 ROS 通讯，build 工作空间后在 /devel/set.bash 文件末尾添加：

主机：

```
export ROS_HOSTNAME=192.168.1.10x（主机ip）
export ROS_MASTER_URI=http://192.168.1.10x（主机ip）:11311
```

从机：

```
export ROS_HOSTNAME=192.168.1.10y（从机ip）
export ROS_MASTER_URI=http://192.168.1.10x（主机ip）:11311
```



#### 运行：

1. 主机、从机、动捕连入同一局域网

2. 主机启动 roscore（确定 source 过上述设置后）

3. 主机 roslaunch vrpn，连接动捕

   ```
   roslaunch vrpn_client_ros sample.launch server:=192.168.1.10z（动捕ip）
   ```

4. 从机启动 cdpr 节点 （确定 source 过上述设置后）

   ```
   rosrun slave cdpr.py
   ```

5. 主机启动控制节点，如果通讯正常，从机终端会显示设置的电机速度



### 关于 CDPR 逆运动学

<img src="/home/xyc/CDPR_dist/note.assets/IK1.png" alt="IK1" style="zoom:80%;" />
<img src="/home/xyc/CDPR_dist/note.assets/IK2.png" alt="IK2" style="zoom:80%;" />
