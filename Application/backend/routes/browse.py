"""
Browse Route

    - GET /api/browse
    - Uses native OS file picker to grab the absolute path of a file or directory
    - Bypasses standard web browser restrictions for localhost tool usage
"""

import threading
import tkinter as tk
from tkinter import filedialog
from flask import Blueprint, request, jsonify

browse_bp = Blueprint("browse", __name__)

def ask_dialog(mode, result):
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        if mode == "directory":
            path = filedialog.askdirectory(title="Select Folder")
        else:
            path = filedialog.askopenfilename(title="Select File")
            
        result['path'] = path
        root.destroy()
    except Exception as e:
        result['path'] = ""
        result['error'] = str(e)

@browse_bp.route("/browse", methods=["GET"])
def browse():
    """
    Triggers the local OS UI file/folder picker
    """
    mode = request.args.get("type", "directory")
    res = {}
    
    # Running in a separate thread so Flask WSGI doesn't block or crash Tkinter
    t = threading.Thread(target=ask_dialog, args=(mode, res))
    t.start()
    t.join()
    
    return jsonify({"path": res.get("path", "")}), 200