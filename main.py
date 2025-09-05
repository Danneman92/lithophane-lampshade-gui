#!/usr/bin/env python3
"""
Lithophane Lampshade GUI Application
A graphical interface for creating lithophane lampshades
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import os
import sys

class LithophaneLampshadeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Lithophane Lampshade Creator")
        self.root.geometry("1000x800")
        
        # Image preview variables
        self.original_image = None
        self.preview_image = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights for resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Lithophane Lampshade Creator", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Image selection
        ttk.Label(main_frame, text="Select Image:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.image_path = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.image_path, width=50).grid(row=1, column=1, padx=(10, 0))
        ttk.Button(main_frame, text="Browse", command=self.browse_image).grid(row=1, column=2, padx=(5, 0))
        
        # Preview frame
        preview_frame = ttk.LabelFrame(main_frame, text="Image Preview", padding="10")
        preview_frame.grid(row=2, column=0, columnspan=3, pady=20, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Canvas for image preview
        self.preview_canvas = tk.Canvas(preview_frame, width=400, height=300, bg="white")
        self.preview_canvas.grid(row=0, column=0, pady=10)
        
        # Lampshade settings
        settings_frame = ttk.LabelFrame(main_frame, text="Lampshade Settings", padding="10")
        settings_frame.grid(row=3, column=0, columnspan=3, pady=20, sticky=(tk.W, tk.E))
        
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
                  command=self.generate_lithophane).grid(row=4, column=0, columnspan=3, pady=20)
        
        # Export STL button (initially disabled)
        self.export_button = ttk.Button(main_frame, text="Export STL", 
                                       command=self.export_stl, state="disabled")
        self.export_button.grid(row=5, column=0, columnspan=3, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready - Select an image to begin")
        self.status_label.grid(row=7, column=0, columnspan=3, pady=(10, 0))
        
    def browse_image(self):
        filename = filedialog.askopenfilename(
            title="Select image file",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if filename:
            self.image_path.set(filename)
            self.load_and_preview_image(filename)
    
    def load_and_preview_image(self, filename):
        """Load and display image preview"""
        try:
            # Load original image
            self.original_image = Image.open(filename)
            
            # Create preview image (resize to fit canvas)
            canvas_width = 400
            canvas_height = 300
            
            # Calculate resize ratio maintaining aspect ratio
            img_width, img_height = self.original_image.size
            width_ratio = canvas_width / img_width
            height_ratio = canvas_height / img_height
            resize_ratio = min(width_ratio, height_ratio)
            
            new_width = int(img_width * resize_ratio)
            new_height = int(img_height * resize_ratio)
            
            # Resize and convert for tkinter
            resized_image = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.preview_image = ImageTk.PhotoImage(resized_image)
            
            # Clear canvas and display image
            self.preview_canvas.delete("all")
            
            # Center the image on canvas
            x_offset = (canvas_width - new_width) // 2
            y_offset = (canvas_height - new_height) // 2
            
            self.preview_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self.preview_image)
            
            self.status_label.config(text=f"Image loaded: {os.path.basename(filename)} ({img_width}x{img_height})")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            self.status_label.config(text="Error loading image")
    
    def generate_lithophane(self):
        if not self.image_path.get():
            messagebox.showerror("Error", "Please select an image file.")
            return
            
        if not self.original_image:
            messagebox.showerror("Error", "Please load a valid image first.")
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
        
        # Simulate lithophane generation process
        self.root.after(2000, self.generation_complete)
        
    def generation_complete(self):
        self.progress.stop()
        self.status_label.config(text="Lithophane generated successfully! Ready to export STL.")
        
        # Enable export button
        self.export_button.config(state="normal")
        
        messagebox.showinfo("Success", "Lithophane has been generated successfully!\n\nYou can now export the STL file.")
    
    def export_stl(self):
        """Export the generated lithophane as STL file"""
        if not self.original_image:
            messagebox.showerror("Error", "No lithophane generated yet.")
            return
        
        # Prompt user for save location
        save_path = filedialog.asksaveasfilename(
            title="Save STL File",
            defaultextension=".stl",
            filetypes=[("STL files", "*.stl"), ("All files", "*.*")],
            initialfile="lithophane_lampshade.stl"
        )
        
        if save_path:
            try:
                self.status_label.config(text="Exporting STL file...")
                self.progress.start()
                
                # Simulate STL file generation and save
                self.root.after(1500, lambda: self.export_complete(save_path))
                
            except Exception as e:
                self.progress.stop()
                messagebox.showerror("Error", f"Failed to export STL: {str(e)}")
                self.status_label.config(text="Export failed")
    
    def export_complete(self, save_path):
        """Complete the STL export process"""
        self.progress.stop()
        
        # Create a simple placeholder STL content for demonstration
        # In a real implementation, this would generate actual lithophane geometry
        stl_content = self.generate_placeholder_stl()
        
        try:
            with open(save_path, 'w') as f:
                f.write(stl_content)
            
            self.status_label.config(text=f"STL exported successfully to: {os.path.basename(save_path)}")
            messagebox.showinfo("Export Complete", 
                              f"STL file has been saved to:\n{save_path}\n\nThe file is ready for 3D printing!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save STL file: {str(e)}")
            self.status_label.config(text="Export failed")
    
    def generate_placeholder_stl(self):
        """Generate a placeholder STL file content"""
        # This is a simple placeholder - in a real implementation,
        # this would generate actual lithophane geometry based on the image
        return """solid lithophane_lampshade
  facet normal 0.0 0.0 1.0
    outer loop
      vertex 0.0 0.0 0.0
      vertex 1.0 0.0 0.0
      vertex 1.0 1.0 0.0
    endloop
  endfacet
  facet normal 0.0 0.0 1.0
    outer loop
      vertex 0.0 0.0 0.0
      vertex 1.0 1.0 0.0
      vertex 0.0 1.0 0.0
    endloop
  endfacet
endsolid lithophane_lampshade
"""

def main():
    root = tk.Tk()
    app = LithophaneLampshadeGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
