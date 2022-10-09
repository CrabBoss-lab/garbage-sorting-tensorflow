from PyQt5.QtWidgets import QApplication
import sys
from QT.rubbish_func import RubbishDlg  # 连接rubbish_func文件


# 应用程序
class RubbishApp(QApplication):
    # 构造方法
    def __init__(self):
        super(RubbishApp,self).__init__(sys.argv)
        # 主窗口
        self.dlg = RubbishDlg()
        # 显示窗口
        self.dlg.show()

