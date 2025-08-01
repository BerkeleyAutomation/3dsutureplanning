#!/usr/bin/env python3
"""
Modern Minimal Image Upload GUI for Suture Planning
Consistent with the progress GUI design
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import os

def set_dpi_awareness_and_font(root):
    import sys
    if sys.platform == "win32":
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    elif sys.platform == "darwin":
        os.environ['TK_SILENCE_DEPRECATION'] = '1'
    try:
        from tkinter import font
        default_font = font.nametofont("TkDefaultFont")
        default_font.configure(family="Arial", size=12)
        root.option_add("*Font", default_font)
    except Exception:
        pass

def center_window(window, width=500, height=300):
    window.update_idletasks()
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    window.geometry(f"{width}x{height}+{x}+{y}")
    window.minsize(width, height)

class ImageUploadGUI:
    """Modern Minimal Image Upload GUI"""
    
    def __init__(self, parent_root=None, title="Image Upload"):
        self.title = title
        self.selected_image_path = None
        self.parent_root = parent_root
        
        # Try to import tkinter
        try:
            import tkinter as tk
            from tkinter import filedialog, messagebox
            TKINTER_AVAILABLE = True
        except ImportError:
            TKINTER_AVAILABLE = False
            print("Warning: tkinter not available.")
            return
        
        if TKINTER_AVAILABLE:
            self._init_gui()
    
    def _init_gui(self):
        """Initialize the GUI components"""
        # Use Toplevel if parent_root is provided and visible, otherwise create new root
        if self.parent_root and self.parent_root.winfo_viewable():
            self.root = tk.Toplevel(self.parent_root)
            self.root.transient(self.parent_root)  # Make it modal to parent
            self.root.grab_set()  # Make it modal
        else:
            self.root = tk.Tk()
        set_dpi_awareness_and_font(self.root)
        self.root.title(self.title)
        center_window(self.root, width=500, height=300)
        self.root.resizable(False, False)
        
        # Configure style - Modern minimal (consistent with progress GUI)
        style = ttk.Style()
        style.theme_use('clam')
        
        # Custom colors (same as progress GUI)
        bg_color = "#f8f9fa"
        accent_color = "#007bff"
        success_color = "#28a745"
        text_color = "#212529"
        
        self.root.configure(bg=bg_color)
        
        # Main container
        main_frame = tk.Frame(self.root, bg=bg_color, padx=30, pady=30)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = tk.Frame(main_frame, bg=bg_color)
        header_frame.pack(fill=tk.X, pady=(0, 30))
        
        title_label = tk.Label(header_frame, text="Suture Planning", 
                              font=('Helvetica', 24, 'bold'), 
                              fg=text_color, bg=bg_color)
        title_label.pack()
        
        subtitle_label = tk.Label(header_frame, text="Image Upload", 
                                 font=('Helvetica', 14), 
                                 fg="#6c757d", bg=bg_color)
        subtitle_label.pack()
        
        # Upload section
        upload_frame = tk.Frame(main_frame, bg=bg_color)
        upload_frame.pack(fill=tk.X, pady=(0, 25))
        
        upload_label = tk.Label(upload_frame, text="Select Wound Image", 
                               font=('Helvetica', 12, 'bold'), 
                               fg=text_color, bg=bg_color)
        upload_label.pack(anchor=tk.W, pady=(0, 15))
        
        # Upload button
        self.upload_button = tk.Button(upload_frame, text="Choose Image File", 
                                      font=('Helvetica', 12, 'bold'),
                                      bg=bg_color, fg="green",
                                      relief=tk.FLAT, padx=20, pady=10,
                                      command=self._browse_file)
        self.upload_button.pack(pady=(0, 15))
        
        # File info
        self.file_info_label = tk.Label(upload_frame, text="No file selected", 
                                       font=('Helvetica', 10), 
                                       fg="#6c757d", bg=bg_color)
        self.file_info_label.pack(anchor=tk.W)
        
        # Action buttons
        button_frame = tk.Frame(main_frame, bg=bg_color)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        # Cancel button
        self.cancel_button = tk.Button(button_frame, text="Cancel", 
                                      font=('Helvetica', 11),
                                      bg="#6c757d", fg="white",
                                      relief=tk.FLAT, padx=20, pady=8,
                                      command=self._cancel)
        self.cancel_button.pack(side=tk.LEFT)
        
        # Start button (hidden initially, will be shown after file selection)
        self.start_button = tk.Button(button_frame, text="Start Suture Planning", 
                                     font=('Helvetica', 11),
                                     bg=success_color, fg="white",
                                     relief=tk.FLAT, padx=20, pady=8,
                                     command=self._start_planning)
        self.start_button.pack(side=tk.RIGHT)
        self.start_button.pack_forget()  # Hide initially
    
    def _browse_file(self):
        """Open file dialog to select image"""
        file_types = [
            ('Image files', '*.jpg *.jpeg *.png *.bmp *.tiff'),
            ('JPEG files', '*.jpg *.jpeg'),
            ('PNG files', '*.png'),
            ('All files', '*.*')
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Wound Image",
            filetypes=file_types
        )
        
        if filename:
            self.selected_image_path = filename
            print(self.selected_image_path)
            self._update_file_info()
            # Automatically close the window after file selection
            self.root.after(500, self._start_planning)  # Small delay to show file info
    
    def _update_file_info(self):
        """Update the file information display"""
        if self.selected_image_path:
            filename = os.path.basename(self.selected_image_path)
            print(filename)
            file_size = os.path.getsize(self.selected_image_path)
            size_mb = file_size / (1024 * 1024)
            
            self.file_info_label.config(
                text=f"Selected: {filename} ({size_mb:.1f} MB)\nStarting suture planning..."
            )
    

            
    def _cancel(self):
        """Cancel the upload and close GUI"""
        self.selected_image_path = None
        self.root.destroy()
    
    def _start_planning(self):
        """Start the suture planning process"""
        self.root.destroy()
    
    def run(self):
        """Run the GUI and return the selected image path"""
        if hasattr(self, 'root'):
            self.root.wait_window()  # Wait for window to be destroyed
            return self.selected_image_path
        else:
            return None

def upload_image_gui(parent_root=None):
    """Convenience function to show image upload GUI"""
    gui = ImageUploadGUI(parent_root)
    return gui.run()

if __name__ == "__main__":
    # Test the GUI
    result = upload_image_gui()
    print(f"Selected image: {result}") 