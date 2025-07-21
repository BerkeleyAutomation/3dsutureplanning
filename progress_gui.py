#!/usr/bin/env python3
"""
Modern Minimal Progress GUI for Suture Planning
Clean, spacious design with subtle colors and excellent readability
"""

import threading
import time
from typing import Optional, Callable
import os

# Try to import tkinter, but provide fallback if not available
try:
    import tkinter as tk
    from tkinter import ttk
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    print("Warning: tkinter not available. Progress GUI will use console output only.")

# Try to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Visualization will be disabled.")

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

def center_window(window, width=1200, height=500):
    window.update_idletasks()
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    window.geometry(f"{width}x{height}+{x}+{y}")
    window.minsize(width, height)

class SutureProgressGUI:
    """Modern Minimal Progress GUI for Suture Planning"""
    
    def __init__(self, title="Suture Planning Progress", parent_root=None):
        self.title = title
        self.is_running = False
        self.current_sutures = 0
        self.total_sutures_range = (0, 0)
        self.best_loss = float('inf')
        self.best_sutures = 0
        
        # Visualization data
        self.current_visualization_data = None
        self.distance_calculator = None
        self.parent_root = parent_root
        
        if TKINTER_AVAILABLE:
            self._init_gui()
        else:
            self._init_console()
    
    def _init_gui(self):
        """Initialize the GUI components"""
        # Use Toplevel if parent_root is provided, otherwise create new root
        if self.parent_root:
            self.root = tk.Toplevel(self.parent_root)
            self.root.transient(self.parent_root)  # Make it modal to parent
            self.root.grab_set()  # Make it modal
        else:
            self.root = tk.Tk()
        set_dpi_awareness_and_font(self.root)
        self.root.title(self.title)
        center_window(self.root, width=1200, height=500)
        self.root.resizable(False, False)
        
        # Configure style - Modern minimal
        style = ttk.Style()
        style.theme_use('clam')
        
        # Custom colors
        bg_color = "#f8f9fa"
        accent_color = "#007bff"
        success_color = "#28a745"
        text_color = "#212529"
        
        self.root.configure(bg=bg_color)
        
        # Main container
        main_frame = tk.Frame(self.root, bg=bg_color, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create left and right panels
        left_panel = tk.Frame(main_frame, bg=bg_color, width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        left_panel.pack_propagate(False)  # Prevent shrinking
        
        right_panel = tk.Frame(main_frame, bg=bg_color)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Header (in left panel)
        header_frame = tk.Frame(left_panel, bg=bg_color)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = tk.Label(header_frame, text="Suture Placement", 
                              font=('Helvetica', 24, 'bold'), 
                              fg=text_color, bg=bg_color)
        title_label.pack()
        
        subtitle_label = tk.Label(header_frame, text="Optimization Progress", 
                                 font=('Helvetica', 14), 
                                 fg="#6c757d", bg=bg_color)
        subtitle_label.pack()
        
        # Overall progress
        progress_frame = tk.Frame(left_panel, bg=bg_color)
        progress_frame.pack(fill=tk.X, pady=(0, 15))
        
        progress_label = tk.Label(progress_frame, text="Overall Progress", 
                                 font=('Helvetica', 12, 'bold'), 
                                 fg=text_color, bg=bg_color)
        progress_label.pack(anchor=tk.W, pady=(0, 8))
        
        # Custom progress bar
        self.progress_canvas = tk.Canvas(progress_frame, height=8, bg="#e9ecef", 
                                       highlightthickness=0, relief=tk.FLAT)
        self.progress_canvas.pack(fill=tk.X, pady=(0, 5))
        
        self.progress_fill = self.progress_canvas.create_rectangle(0, 0, 0, 8, 
                                                                 fill=accent_color, outline="")
        
        self.progress_text = tk.Label(progress_frame, text="0%", 
                                     font=('Helvetica', 10), 
                                     fg="#6c757d", bg=bg_color)
        self.progress_text.pack(anchor=tk.W)
        
        # Current optimization
        current_frame = tk.Frame(left_panel, bg=bg_color)
        current_frame.pack(fill=tk.X, pady=(0, 15))
        
        current_label = tk.Label(current_frame, text="Current Optimization", 
                                font=('Helvetica', 12, 'bold'), 
                                fg=text_color, bg=bg_color)
        current_label.pack(anchor=tk.W, pady=(0, 8))
        
        # Suture info
        self.suture_info = tk.Label(current_frame, text="Testing: 0 sutures", 
                                   font=('Helvetica', 16), 
                                   fg=accent_color, bg=bg_color)
        self.suture_info.pack(anchor=tk.W, pady=(0, 15))
        
        # Loss values in a grid
        loss_frame = tk.Frame(current_frame, bg=bg_color)
        loss_frame.pack(fill=tk.X)
        
        self.total_loss_label = tk.Label(loss_frame, text="Total Loss: --", 
                                        font=('Helvetica', 11), 
                                        fg=text_color, bg=bg_color)
        self.total_loss_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 30))
        
        self.closure_loss_label = tk.Label(loss_frame, text="Closure: --", 
                                          font=('Helvetica', 11), 
                                          fg=text_color, bg=bg_color)
        self.closure_loss_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 30))
        
        self.shear_loss_label = tk.Label(loss_frame, text="Shear: --", 
                                        font=('Helvetica', 11), 
                                        fg=text_color, bg=bg_color)
        self.shear_loss_label.grid(row=0, column=2, sticky=tk.W)
        
        # Best result
        best_frame = tk.Frame(left_panel, bg=bg_color)
        best_frame.pack(fill=tk.X, pady=(0, 15))
        
        best_label = tk.Label(best_frame, text="Best Result", 
                             font=('Helvetica', 12, 'bold'), 
                             fg=text_color, bg=bg_color)
        best_label.pack(anchor=tk.W, pady=(0, 8))
        
        self.best_info = tk.Label(best_frame, text="--", 
                                 font=('Helvetica', 14), 
                                 fg=success_color, bg=bg_color)
        self.best_info.pack(anchor=tk.W)
        
        # Status
        status_frame = tk.Frame(left_panel, bg=bg_color)
        status_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.status_label = tk.Label(status_frame, text="Ready to start...", 
                                    font=('Helvetica', 11), 
                                    fg="#6c757d", bg=bg_color)
        self.status_label.pack(anchor=tk.W)
        
        # Visualization area (right panel)
        if MATPLOTLIB_AVAILABLE:
            viz_frame = tk.Frame(right_panel, bg=bg_color)
            viz_frame.pack(fill=tk.BOTH, expand=True, padx=(0, 0))
            
            viz_label = tk.Label(viz_frame, text="Current Suture Plan", 
                                font=('Helvetica', 14, 'bold'), 
                                fg=text_color, bg=bg_color)
            viz_label.pack(anchor=tk.W, pady=(0, 10))
            
            # Create matplotlib figure - bigger for right panel
            self.fig = Figure(figsize=(10, 7), dpi=80, facecolor=bg_color)
            self.ax = self.fig.add_subplot(111)
            self.ax.set_facecolor(bg_color)
            
            # Create canvas
            self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Initial plot
            self.ax.text(0.5, 0.5, 'Waiting for optimization...', 
                        ha='center', va='center', transform=self.ax.transAxes,
                        fontsize=14, color='#6c757d')
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
            self.ax.axis('off')
            self.canvas.draw()
        else:
            # Placeholder for when matplotlib is not available
            viz_frame = tk.Frame(right_panel, bg=bg_color)
            viz_frame.pack(fill=tk.BOTH, expand=True, padx=(10, 0))
            
            viz_label = tk.Label(viz_frame, text="Current Suture Plan", 
                                font=('Helvetica', 14, 'bold'), 
                                fg=text_color, bg=bg_color)
            viz_label.pack(anchor=tk.W, pady=(0, 10))
            
            placeholder_label = tk.Label(viz_frame, text="Visualization not available\n(requires matplotlib)", 
                                       font=('Helvetica', 12), 
                                       fg="#6c757d", bg=bg_color, justify=tk.CENTER)
            placeholder_label.pack(fill=tk.BOTH, expand=True)
    
    def _init_console(self):
        """Initialize console-only mode"""
        self.root = None
        print(f"\n=== {self.title} ===")
        print("Console mode - GUI not available")
        print("=" * 50)
    
    def start(self):
        """Start the GUI in the main thread"""
        self.is_running = True
        if TKINTER_AVAILABLE and self.root:
            pass  # No animation needed for this design
        else:
            print("Starting optimization...")
    
    def _run_gui(self):
        """Run the GUI main loop"""
        if TKINTER_AVAILABLE and self.root:
            self.root.mainloop()
    
    def stop(self):
        """Stop the GUI"""
        self.is_running = False
        if TKINTER_AVAILABLE and self.root:
            self.root.quit()
            self.root.destroy()
        else:
            print("Optimization complete!")
    
    def update_overall_progress(self, progress: float, stage: str = ""):
        """Update the overall progress bar (0.0 to 1.0)"""
        if not self.is_running:
            return
        
        if TKINTER_AVAILABLE and self.root:
            # Update progress bar
            width = self.progress_canvas.winfo_width()
            fill_width = int(width * progress)
            self.progress_canvas.coords(self.progress_fill, 0, 0, fill_width, 8)
            self.progress_text.config(text=f"{progress*100:.1f}%")
            self.root.update()
        else:
            print(f"Progress: {progress*100:.1f}% - {stage}")
    
    def set_suture_range(self, start_range: int, end_range: int):
        """Set the range of sutures being tested"""
        self.total_sutures_range = (start_range, end_range)
    
    def update_current_sutures(self, num_sutures: int):
        """Update the current number of sutures being optimized"""
        if not self.is_running:
            return
        
        self.current_sutures = num_sutures
        
        if TKINTER_AVAILABLE and self.root:
            self.suture_info.config(text=f"Testing: {num_sutures} sutures")
            if self.total_sutures_range[1] > 0:
                progress = (num_sutures - self.total_sutures_range[0]) / (self.total_sutures_range[1] - self.total_sutures_range[0] + 1)
                self.update_overall_progress(progress)
            self.root.update()
        else:
            if self.total_sutures_range[1] > 0:
                progress = (num_sutures - self.total_sutures_range[0]) / (self.total_sutures_range[1] - self.total_sutures_range[0] + 1)
                print(f"Testing sutures: {num_sutures} (Progress: {progress*100:.1f}%)")
            else:
                print(f"Testing sutures: {num_sutures}")
    
    def update_losses(self, total_loss: float, closure_loss: float, shear_loss: float, 
                     center_var_loss: float = None, ins_ext_var_loss: float = None, ideal_loss: float = None):
        """Update the loss values display"""
        if not self.is_running:
            return
        
        if TKINTER_AVAILABLE and self.root:
            self.total_loss_label.config(text=f"Total Loss: {total_loss:.4f}")
            self.closure_loss_label.config(text=f"Closure: {closure_loss:.4f}")
            self.shear_loss_label.config(text=f"Shear: {shear_loss:.4f}")
            
            if total_loss < self.best_loss:
                self.best_loss = total_loss
                self.best_sutures = self.current_sutures
                self.best_info.config(text=f"{self.best_sutures} sutures - Loss: {self.best_loss:.4f}")
            
            self.root.update()
        else:
            print(f"  Total Loss: {total_loss:.4f}")
            print(f"  Closure Loss: {closure_loss:.4f}")
            print(f"  Shear Loss: {shear_loss:.4f}")
            
            if total_loss < self.best_loss:
                self.best_loss = total_loss
                self.best_sutures = self.current_sutures
                print(f"  *** NEW BEST: {self.best_sutures} sutures with loss {self.best_loss:.4f} ***")
    
    def set_status(self, status: str):
        """Update the status message"""
        if not self.is_running:
            return
        
        if TKINTER_AVAILABLE and self.root:
            self.status_label.config(text=status)
            self.root.update()
        else:
            print(f"Status: {status}")
    
    def set_distance_calculator(self, distance_calculator):
        """Set the distance calculator for visualization"""
        self.distance_calculator = distance_calculator
    
    def update_visualization(self, wound_point_t, title="Current Suture Plan"):
        """Update the visualization with current suture data"""
        if not self.is_running or not MATPLOTLIB_AVAILABLE or not self.distance_calculator:
            return
        
        if TKINTER_AVAILABLE and self.root:
            try:
                # Clear the current plot
                self.ax.clear()
                self.ax.set_facecolor("#f8f9fa")
                
                # Get the wound curve points for plotting
                num_pts = len(wound_point_t)
                
                # Get the curve and gradient for each point
                wound_points, wound_curve = self.distance_calculator.wound_parametric(wound_point_t, 0)
                wound_derivatives_x, wound_derivatives_y = self.distance_calculator.wound_parametric(wound_point_t, 1)
                
                # Calculate insertion and extraction points
                import math
                def get_norm(x, y):
                    return math.sqrt(x**2 + y**2)
                
                norms = [get_norm(wound_derivatives_x[i], wound_derivatives_y[i]) for i in range(len(wound_derivatives_x))]
                normal_vecs = [[wound_derivatives_y[i]/norms[i], -wound_derivatives_x[i]/norms[i]] for i in range(num_pts)]
                normal_vecs = [[normal_vec[0] * self.distance_calculator.wound_width, normal_vec[1] * self.distance_calculator.wound_width] for normal_vec in normal_vecs]
                
                insert_pts = [[wound_points[i] + normal_vecs[i][0], wound_curve[i] + normal_vecs[i][1]] for i in range(num_pts)]
                extract_pts = [[wound_points[i] - normal_vecs[i][0], wound_curve[i] - normal_vecs[i][1]] for i in range(num_pts)]
                center_pts = [[wound_points[i], wound_curve[i]] for i in range(num_pts)]
                
                # Plot the wound curve
                X_, Y_ = [], []
                for i in range(500):
                    t = min(wound_point_t) + (max(wound_point_t) - min(wound_point_t))*i/500
                    temp = self.distance_calculator.wound_parametric(t, 0)
                    X_.append(temp[1])
                    Y_.append(-temp[0])
                
                self.ax.plot(X_, Y_, color='black', linewidth=1.5, alpha=0.7)
                
                # Plot suture points
                self.ax.scatter([insert_pts[i][1] for i in range(num_pts)], 
                              [-insert_pts[i][0] for i in range(num_pts)], 
                              c="red", s=30, alpha=0.8, label='Insertion')
                self.ax.scatter([extract_pts[i][1] for i in range(num_pts)], 
                              [-extract_pts[i][0] for i in range(num_pts)], 
                              c="blue", s=30, alpha=0.8, label='Extraction')
                self.ax.scatter([center_pts[i][1] for i in range(num_pts)], 
                              [-center_pts[i][0] for i in range(num_pts)], 
                              c="green", s=30, alpha=0.8, label='Center')
                
                # Draw suture lines
                for i in range(len(insert_pts)):
                    self.ax.plot([insert_pts[i][1], extract_pts[i][1]], 
                               [-insert_pts[i][0], -extract_pts[i][0]], 
                               color='black', linewidth=1, alpha=0.6)
                
                # Set title and formatting
                self.ax.set_title(f"{title}\n{self.current_sutures} sutures - Loss: {self.best_loss:.4f}", 
                                fontsize=11, color='#212529', pad=10)
                self.ax.axis('square')
                self.ax.grid(True, alpha=0.2)
                self.ax.legend(loc='upper right', fontsize=8)
                
                # Remove axis labels for cleaner look
                self.ax.set_xlabel('')
                self.ax.set_ylabel('')
                
                # Update the canvas
                self.canvas.draw()
                self.root.update()
                
            except Exception as e:
                print(f"Error updating visualization: {e}")
                # Fallback to simple text
                self.ax.clear()
                self.ax.set_facecolor("#f8f9fa")
                self.ax.text(0.5, 0.5, f'Current: {self.current_sutures} sutures\nLoss: {self.best_loss:.4f}', 
                            ha='center', va='center', transform=self.ax.transAxes,
                            fontsize=12, color='#6c757d')
                self.ax.set_xlim(0, 1)
                self.ax.set_ylim(0, 1)
                self.ax.axis('off')
                self.canvas.draw()
    
    def mark_complete(self):
        """Mark the optimization as complete"""
        if not self.is_running:
            return
        
        if TKINTER_AVAILABLE and self.root:
            self.update_overall_progress(1.0)
            self.status_label.config(text="Optimization complete!")
            self.root.update()
        else:
            print("Optimization complete!")
            if self.best_sutures > 0:
                print(f"Best result: {self.best_sutures} sutures with loss {self.best_loss:.4f}")

# Global instance for easy access
_progress_gui = None

def start_progress_gui(title="Suture Planning Progress", parent_root=None):
    """Start the progress GUI"""
    global _progress_gui
    _progress_gui = SutureProgressGUI(title, parent_root)
    _progress_gui.start()
    return _progress_gui

def stop_progress_gui():
    """Stop the progress GUI"""
    global _progress_gui
    if _progress_gui:
        _progress_gui.stop()
        _progress_gui = None

def get_progress_gui():
    """Get the current progress GUI instance"""
    global _progress_gui
    return _progress_gui 