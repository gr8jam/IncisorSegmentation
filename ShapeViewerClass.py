from matplotlib import pyplot as plt
import myLib
import numpy as np


class ShapeHandle:
    def __init__(self, shape_color='b', linewidth=1):
        # self.lm_org = None
        # self.lm_loc = plt.plot([], [], color=shape_color, marker='.', markersize=5, linestyle=' ')[0]
        self.start = plt.plot([], [], color='c', marker='.', markersize=1, linestyle=' ')[0]
        self.border = plt.plot([], [], color=shape_color, linestyle='-', linewidth=linewidth)[0]

        # self.lm_org = plt.plot([], [], color=shape_color, marker='.', markersize=5)[0]
        # self.lm_org_start = plt.plot([], [], color='c', marker='.', markersize=5)[0]
        # self.center = plt.plot([], [], color=shape_color, marker='.', markersize=6)[0]
        # self.profile = plt.plot([], [], color='c', marker='.', markersize=5)[0]


class ShapesViewer:
    def __init__(self, shapes_list, shapes_ref, title=""):
        # Configure figure with local shape
        self.fig_loc = plt.figure()
        self.axes = plt.gca()
        myLib.move_figure('top-left')  # manual_position=[800, 100, 400, 800])
        plt.axis('equal')
        # plt.grid()
        plt.title(title)

        # Configure figure with original shape


        self.handle_list = []
        self.shapes_list = shapes_list

        for shape in self.shapes_list:
            handle = ShapeHandle()
            self.handle_list.append(handle)

        self.handle_ref = ShapeHandle('r', linewidth=3)
        self.shapes_ref = shapes_ref

    def update_shape(self, handle, shape):
        s = shape.scale
        s = 1
        # Update landmarks in local coordinate system
        # handle.lm_loc.set_xdata(shape.lm_loc[0, :] * s)
        # handle.lm_loc.set_ydata(shape.lm_loc[1, :] * s)

        # Update border
        handle.border.set_xdata(shape.lm_loc[0, :] * s)
        handle.border.set_ydata(shape.lm_loc[1, :] * s)

        # Update first landmark position
        handle.start.set_xdata(shape.lm_loc[0, 0] * s)
        handle.start.set_ydata(shape.lm_loc[1, 0] * s)

        # Update center
        # self.handle.center.set_xdata(shape.center[0, :])
        # self.handle.center.set_ydata(shape.center[1, :])

        # Update profile coordinates
        # handle.profile.set_xdata(shape.profile_coordinates[0, :, 0])
        # handle.profile.set_ydata(shape.profile_coordinates[1, :, 0])

        # recompute the ax.dataLim
        self.axes.relim()
        # update ax.viewLim using the new dataLim
        self.axes.autoscale_view()
        plt.draw()
        plt.show(block=False)

    def update_shapes_ref(self):
        self.update_shape(self.handle_ref, self.shapes_ref)

    def update_shapes_all(self):
        self.update_shapes_ref()
        for idx in range(len(self.handle_list)):
            self.update_shape(self.handle_list[idx], self.shapes_list[idx])
            # plt.waitforbuttonpress()
            # self.update_shapes_ref()
            # print self.shapes_list[idx].ssd

    def update_shape_idx(self, shape_idx):
        self.update_shape(self.handle_list[shape_idx], self.shapes_list[shape_idx])

    def set_visible_profiles(self, visible=True):
        for handle in self.handle_list:
            handle.profile.set_visible(visible)
