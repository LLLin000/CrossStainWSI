import tkinter as tk
import pytest

from crossstainwsi.gui.app import CrossStainWSIGUI


def test_gui_initialization():
    try:
        root = tk.Tk()
        # 隐藏窗口避免弹出干扰测试
        root.withdraw()
        app = CrossStainWSIGUI(root)
        assert app.root == root
        assert app.wsi_dir_var.get() != ""
        root.destroy()
    except tk.TclError:
        # 在无显示环境 (无头 CI) 中跳过
        pytest.skip("No graphical display available for Tkinter GUI test")
