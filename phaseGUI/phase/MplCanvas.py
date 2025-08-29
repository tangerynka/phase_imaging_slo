from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar


class MplCanvasSingle(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100, image=True):
        self.fig =plt.figure()
        self.image = image
        self.clear()
        super(MplCanvasSingle, self).__init__(self.fig)

    def clear(self):
        self.fig.clear()
        self.axes = self.fig.add_subplot(111)
        if self.image:
            self.axes.set_axis_off()
            self.axes.get_xaxis().set_visible(False)
            self.axes.get_yaxis().set_visible(False)
            self.axes.set_xticks([])
            self.axes.set_yticks([])
            for spine in self.axes.spines.values():
                spine.set_visible(False)
            self.fig.tight_layout(pad=0)
        else:
            self.fig.set_layout_engine('constrained')


class MplCanvasDouble(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig =plt.figure()
        self.clear()
        super(MplCanvasDouble, self).__init__(self.fig)
    
    def clear(self):
        self.fig.clear()
        self.axes1 = self.fig.add_subplot(121)
        self.axes1.set_axis_off()
        self.axes2 = self.fig.add_subplot(122)
        self.axes2.set_axis_off()
        self.axes1.sharex(self.axes2)
        self.axes1.sharey(self.axes2)
        self.fig.tight_layout(pad=0)