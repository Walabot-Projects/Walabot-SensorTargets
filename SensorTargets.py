from __future__ import print_function, division
from math import sin, radians
import WalabotAPI as wlbt
try: # for Python 2
    import Tkinter as tk
except ImportError: # for Python 3
    import tkinter as tk
try:
    range = xrange
except NameError:
    pass

APP_X, APP_Y = 50, 50 # location of top-left corner of window
CANVAS_LENGTH = 500


class SensorTargetsApp(tk.Frame):

    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.leftPanel = LeftPanel(self)
        self.rightPanel = RightPanel(self)
        self.leftPanel.pack(side=tk.LEFT)
        self.rightPanel.pack(side=tk.RIGHT)


class LeftPanel(tk.Frame):

    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        #self.wlbtPanel = WalabotPanel(self)
        #self.ctrlPanel = ControlPanel(self)
        #self.wlbtPanel.pack(side=tk.TOP)
        #self.ctrlPanel.pack(side=tk.TOP)


class RightPanel(tk.Frame):

    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.canvasPanel = CanvasPanel(self)
        self.canvasPanel.pack()


class CanvasPanel(tk.LabelFrame):

    def __init__(self, parent):
        tk.LabelFrame.__init__(self, parent, text="Sensor Targets")
        self.targetsCanvas = TargetsCanvas(self)
        self.targetsCanvas.pack()


class TargetsCanvas(tk.Canvas):

    def __init__(self, parent):
        tk.Canvas.__init__(self, parent, width=CANVAS_LENGTH, height=CANVAS_LENGTH)


def configureWindow(root):
    root.title("Walabot - Sensor Targets")
    iconFile = tk.PhotoImage(file="walabot-icon.gif")
    root.tk.call("wm", "iconphoto", root._w, iconFile) # set app icon
    root.geometry("+{}+{}".format(APP_X, APP_Y)) # set window location
    root.option_add("*Font", "TkFixedFont")
    root.update()
    root.minsize(width=root.winfo_reqwidth(), height=root.winfo_reqheight())

def sensorTargets():
    root = tk.Tk()
    SensorTargetsApp(root).pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    configureWindow(root)
    root.mainloop()

if __name__ == "__main__":
    sensorTargets()
