"""
Tkinter Test
    - Tests Tkinter file dialog
"""

import threading
import tkinter as tk
from tkinter import filedialog

def ask_dir(result):
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        result['path'] = filedialog.askdirectory(title="Test Folder Selection")
        root.destroy()
    except Exception as e:
        result['error'] = str(e)

res = {}
t = threading.Thread(target=ask_dir, args=(res,))
t.start()
t.join()
print("Result:", res)