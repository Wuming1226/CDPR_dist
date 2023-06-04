from simple_pid import PID


class controller(object):
    def __init__(self, p, i, d, output_limit: tuple):
        self.controller = PID(p, i, d, output_limits=output_limit)

    def set_point(self, point):
        self.controller.setpoint = point

    def get_output(self, value):
        output = self.controller(value)
        return output


if __name__ == "__main__":
    pid = controller(1, 0.1, 0.1, (-0.3, 0.3))
    pid.set_point(1)
    while True:
        output = pid.get_output(0)
        print(output)
