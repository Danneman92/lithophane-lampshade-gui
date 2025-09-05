#!/usr/bin/env python3
"""
Lithophane Lampshade GUI Application
A graphical interface for creating lithophane lampshades
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys

class LithophaneLampshadeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Lithophane Lampshade Creator")
        self.root.geometry("800x600")
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Lithophane Lampshade Creator", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Image selection
        ttk.Label(main_frame, text="Select Image:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.image_path = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.image_path, width=50).grid(row=1, column=1, padx=(10, 0))
        ttk.Button(main_frame, text="Browse", command=self.browse_image).grid(row=1, column=2, padx=(5, 0))
        
        # Lampshade settings
        settings_frame = ttk.LabelFrame(main_frame, text="Lampshade Settings", padding="10")
        settings_frame.grid(row=2, column=0, columnspan=3, pady=20, sticky=(tk.W, tk.E))
        
        # Diameter
        ttk.Label(settings_frame, text="Diameter (mm):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.diameter = tk.StringVar(value="150")
        ttk.Entry(settings_frame, textvariable=self.diameter, width=10).grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # Height
        ttk.Label(settings_frame, text="Height (mm):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.height = tk.StringVar(value="200")
        ttk.Entry(settings_frame, textvariable=self.height, width=10).grid(row=1, column=1, sticky=tk.W, padx=(10, 0))
        
        # Thickness
        ttk.Label(settings_frame, text="Thickness (mm):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.thickness = tk.StringVar(value="2.5")
        ttk.Entry(settings_frame, textvariable=self.thickness, width=10).grid(row=2, column=1, sticky=tk.W, padx=(10, 0))
        
        # Generate button
        ttk.Button(main_frame, text="Generate Lithophane", 
                  command=self.generate_lithophane).grid(row=3, column=0, columnspan=3, pady=20)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready")
        self.status_label.grid(row=5, column=0, columnspan=3, pady=(10, 0))
        
    def browse_image(self):
        filename = filedialog.askopenfilename(
            title="Select image file",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if filename:
            self.image_path.set(filename)
    
    def generate_lithophane(self):
        if not self.image_path.get():
            messagebox.showerror("Error", "Please select an image file.")
            return
            
        try:
            diameter = float(self.diameter.get())
            height = float(self.height.get())
            thickness = float(self.thickness.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values for dimensions.")
            return
            
        self.status_label.config(text="Generating lithophane...")
        self.progress.start()
        
        # Placeholder for actual lithophane generation logic
        self.root.after(3000, self.generation_complete)
        
    def generation_complete(self):
        self.progress.stop()
        self.status_label.config(text="Lithophane generated successfully!")
        messagebox.showinfo("Success", "Lithophane has been generated successfully!")

def main():
    root = tk.Tk()
    app = LithophaneLampshadeGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
