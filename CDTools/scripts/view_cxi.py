from __future__ import division, print_function, absolute_import
import h5py
import numpy as np
import math
import os
from PyQt5 import QtWidgets
from matplotlib import pyplot as plt
import signal
import sys
import argparse

#
# The plotting was broken by the move to CDTools, at some point this should
# be fixed
#

def view_cxi(filename):
    """Opens a popup window displaying the contents of ``filename``. 
    relevant attributes."""
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    class Viewer(QtWidgets.QMainWindow):
        def __init__(self, datafile):
            self.datafile = datafile
            QtWidgets.QMainWindow.__init__(self)
            self.tree = QtWidgets.QTreeWidget(self)
            self.tree.setColumnWidth(0, 200)
            self.setCentralWidget(self.tree)
            self.buildTree()
            self.tree.itemClicked.connect(self.handleClick)

        def handleClick(self,item,column):
            if(item.text(column) == 'Click to print to console'):
                data = self.data_full[str(item.text(2))]
                print(np.asarray(data))
            if(item.text(column) == 'Click to display'):
                data = self.datasets[str(item.text(2))]
                plt.plot(data)
                plt.title(item.text(0).capitalize())
                plt.show()

        def closeWindow(self):
            pass

        def buildTree(self):
            self.datasets = {}
            self.data_full = {}
            self.tree.setColumnCount(2)
            self.f = h5py.File(self.datafile, 'r')
            item = QtWidgets.QTreeWidgetItem(['/'])
            self.tree.addTopLevelItem(item)
            self.buildBranch(self.f,item)

        def buildBranch(self,group,item):
            for g in group.keys():
                lst = [g]
                self.data_full[group[g].name] = group[g]
                if(isinstance(group[g],h5py.Group)):
                    child = QtWidgets.QTreeWidgetItem(lst)
                    self.buildBranch(group[g],child)
                    item.addChild(child)
                else:
                    if len(group[g].shape)>2:
                        lst.append('Click to print to console')
                        lst.append(group[g].name)
                        self.datasets[group[g].name] = group[g]
                        item.addChild(QtWidgets.QTreeWidgetItem(lst))
                    if len(group[g].shape)==2 or len(group[g].shape)==1:
                        lst.append('Click to display')
                        lst.append(group[g].name)
                        self.datasets[group[g].name] = group[g]
                        item.addChild(QtWidgets.QTreeWidgetItem(lst))
                    else:
                        lst.append('Click to print to console')
                        lst.append(group[g].name)
                        self.datasets[group[g].name] = group[g]
                        item.addChild(QtWidgets.QTreeWidgetItem(lst))

    filename = os.path.expanduser(filename)
    app = QtWidgets.QApplication(sys.argv)
    viewer = Viewer(filename)
    viewer.setFixedSize(500, 500)
    viewer.show()
    app.exec_()


def make_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument('file', help='The cxi file to view')
    return parser


if __name__ == '__main__':
    args = make_argparser().parse_args()

    view_cxi(args.file)
