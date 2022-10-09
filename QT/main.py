# Author:yujunyu
# -*- codeing = utf-8 -*-
# @Time :2022/8/19 16:02
# @Author :yujunyu
# @Site :
# @File :main.py
# @software: PyCharm

import sys
from QT.rubbish_app import RubbishApp
if __name__ == '__main__':
    app = RubbishApp()
    status = app.exec()
    sys.exit(app.exec())
