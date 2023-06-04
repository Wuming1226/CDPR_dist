import numpy as np
import matplotlib.pyplot as plt

# origin offset
xOff = 0.478
yOff = 0.425
zOff = -0.340

T = 0.1
for i in range(1, 2):
    file_order = '1000_0{}'.format(i)
    file_order = '10hz1005_6'
    # X_path = 'sim_origin_data/5hz/' + file_order + '_X.txt'
    # U_path = 'sim_origin_data/5hz/' + file_order + '_U.txt'
    # dX_path = 'sim_origin_data/5hz/' + file_order + '_dX.txt'
    X_path = 'traject_data/0508/' + file_order + '_X.txt'

    X_data_list = np.loadtxt(X_path, dtype='float')
    x_list = X_data_list[:, 0]
    y_list = X_data_list[:, 1]
    z_list = X_data_list[:, 2]

    x_r_list, y_r_list, z_r_list = [], [], []

    for cnt in range(400):
        x_r_list.append(0.10 * np.sin(np.pi/20 * (cnt * T)))
        y_r_list.append(0.10 * np.cos(np.pi/20 * (cnt * T)))
        z_r_list.append(0.025 * np.sin(np.pi/10 * (cnt * T)))

    x_list = x_list[1000:] - xOff
    y_list = y_list[1000:] - yOff
    z_list = z_list[1000:] - zOff

    select = np.arange(1, 401, 5)
    x_list = x_list[select]
    y_list = y_list[select]
    z_list = z_list[select]

    # plot
    # fig = plt.figure(i)
    # x_plot = fig.add_subplot(3,1,1)
    # y_plot = fig.add_subplot(3,1,2)
    # z_plot = fig.add_subplot(3,1,3)
    # plt.ion()

    # x_plot.plot(x_r_list)
    # x_plot.plot(x_list)
    # y_plot.plot(y_r_list)
    # y_plot.plot(y_list)
    # z_plot.plot(z_r_list)
    # z_plot.plot(z_list)

    # x_plot.set_ylabel('x')
    # y_plot.set_ylabel('y')
    # z_plot.set_ylabel('z')

    fig2 = plt.figure(i+6)
    traject_plot = fig2.add_subplot(1,1,1)
    traject_plot.plot(x_r_list, y_r_list)
    traject_plot.plot(x_list, y_list)

plt.ioff()
plt.show()