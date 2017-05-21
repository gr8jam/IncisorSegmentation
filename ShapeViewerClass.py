from matplotlib import pyplot as plt
import myLib

class ShapeHandle:
    def __init__(self):
        # self.lm_org = None
        self.lm_loc = None
        self.border = None
        self.center = None


class ShapesViewer:
    def __init__(self, shapes_list, shapes_ref):
        self.fig = plt.figure()
        self.axes = plt.gca()
        myLib.move_figure('top-right')
        plt.axis('equal')
        plt.grid()

        self.handle_ref = ShapeHandle()
        self.handle_ref.lm_loc = plt.plot([], [], 'k.', markersize=10)[0]
        self.handle_ref.border = plt.plot([], [], 'b-', linewidth=3)[0]
        # self.handle_ref.center = plt.plot([], [], 'b.', markersize=10)[0]
        self.shapes_ref = shapes_ref

        self.handle_list = []
        self.shapes_list = shapes_list

        for shape in self.shapes_list:
            handle = ShapeHandle()
            handle.lm_loc = plt.plot([], [], 'k.', markersize=10)[0]
            handle.border = plt.plot([], [], 'r--')[0]
            # handle.lm_loc = plt.plot(shape.lm_loc[0, :], shape.lm_org[1, :], 'k.', markersize=20)[0]
            # handle.border = plt.plot(shape.lm_loc[0, :], shape.lm_loc[1, :], 'r--')[0]
            # handle.center = plt.plot(shape.lm_org[0, :], shape.lm_org[1, :], 'r.', markersize=10)[0]
            self.handle_list.append(handle)

    def update_shape(self, handle, shape):
        # Update landmarks in local coordinate system
        handle.lm_loc.set_xdata(shape.lm_loc[0, :])
        handle.lm_loc.set_ydata(shape.lm_loc[1, :])

        # Update border
        handle.border.set_xdata(shape.lm_loc[0, :])
        handle.border.set_ydata(shape.lm_loc[1, :])

        # Update center
        # self.handle.center.set_xdata(shape.center[0, :])
        # self.handle.center.set_ydata(shape.center[1, :])

        # recompute the ax.dataLim
        self.axes.relim()
        # update ax.viewLim using the new dataLim
        self.axes.autoscale_view()
        plt.draw()
        plt.show(block=False)

    def update_shapes_ref(self):
        self.update_shape(self.handle_ref, self.shapes_ref)

    def update_shapes_all(self):
        for idx in range(len(self.handle_list)):
            self.update_shape(self.handle_list[idx], self.shapes_list[idx])
            # plt.waitforbuttonpress()
            # self.update_shapes_ref()
