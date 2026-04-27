#!/usr/bin/env python3
import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt
from main_window import MainWindow


def _apply_dark_palette(app: QApplication):
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window,          QColor(32,  33,  36))
    pal.setColor(QPalette.WindowText,      QColor(232, 234, 237))
    pal.setColor(QPalette.Base,            QColor(27,  27,  27))
    pal.setColor(QPalette.AlternateBase,   QColor(45,  45,  45))
    pal.setColor(QPalette.ToolTipBase,     QColor(33,  34,  37))
    pal.setColor(QPalette.ToolTipText,     QColor(232, 234, 237))
    pal.setColor(QPalette.Text,            QColor(232, 234, 237))
    pal.setColor(QPalette.Button,          QColor(60,  64,  67))
    pal.setColor(QPalette.ButtonText,      QColor(232, 234, 237))
    pal.setColor(QPalette.BrightText,      Qt.red)
    pal.setColor(QPalette.Link,            QColor(138, 180, 248))
    pal.setColor(QPalette.Highlight,       QColor(66,  133, 244))
    pal.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(pal)


def main():
    app = QApplication(sys.argv)
    _apply_dark_palette(app)
    qss_path = os.path.join(os.path.dirname(__file__), "style.qss")
    if os.path.isfile(qss_path):
        try:
            with open(qss_path, "r", encoding="utf-8") as f:
                app.setStyleSheet(f.read())
        except Exception:
            pass
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
