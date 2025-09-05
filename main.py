#!/usr/bin/env python3
"""
Lithophane Lampshade GUI Application
A graphical interface for creating lithophane lampshades
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
from stl import mesh

class LithophaneLampshadeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Lithophane Lampshade Creator")
        self.root.geometry("1200x900")
        
        # Image and parameters
        self.original_image = None
        self.preview_image = None
        self.height = tk.DoubleVar(value=100.0)
        self.diameter = tk.DoubleVar(value=80.0)
        self.thickness = tk.DoubleVar(value=3.0)
        self.min_thickness = tk.DoubleVar(value=0.8)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Lithophane Lampshade Creator", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Left panel for controls
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Image selection
        ttk.Label(left_frame, text="Image Selection", font=('Arial', 12, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        ttk.Button(left_frame, text="Select Image", command=self.select_image).grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Parameters frame
        params_frame = ttk.LabelFrame(left_frame, text="Lampshade Parameters", padding="10")
        params_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Height parameter
        ttk.Label(params_frame, text="Height (mm):").grid(row=0, column=0, sticky=tk.W)
        height_scale = ttk.Scale(params_frame, from_=50, to=200, variable=self.height, orient=tk.HORIZONTAL)
        height_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
        ttk.Label(params_frame, textvariable=self.height).grid(row=0, column=2, padx=(10, 0))
        
        # Diameter parameter
        ttk.Label(params_frame, text="Diameter (mm):").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        diameter_scale = ttk.Scale(params_frame, from_=40, to=150, variable=self.diameter, orient=tk.HORIZONTAL)
        diameter_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(5, 0))
        ttk.Label(params_frame, textvariable=self.diameter).grid(row=1, column=2, padx=(10, 0), pady=(5, 0))
        
        # Thickness parameter
        ttk.Label(params_frame, text="Max Thickness (mm):").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        thickness_scale = ttk.Scale(params_frame, from_=1, to=5, variable=self.thickness, orient=tk.HORIZONTAL)
        thickness_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(5, 0))
        ttk.Label(params_frame, textvariable=self.thickness).grid(row=2, column=2, padx=(10, 0), pady=(5, 0))
        
        # Min thickness parameter
        ttk.Label(params_frame, text="Min Thickness (mm):").grid(row=3, column=0, sticky=tk.W, pady=(5, 0))
        min_thickness_scale = ttk.Scale(params_frame, from_=0.4, to=2, variable=self.min_thickness, orient=tk.HORIZONTAL)
        min_thickness_scale.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(5, 0))
        ttk.Label(params_frame, textvariable=self.min_thickness).grid(row=3, column=2, padx=(10, 0), pady=(5, 0))
        
        params_frame.columnconfigure(1, weight=1)
        
        # Generate button
        ttk.Button(left_frame, text="Generate Lampshade STL", command=self.generate_lampshade).grid(row=3, column=0, sticky=(tk.W, tk.E), pady=10)
        
        # Right panel for image preview
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Image preview
        self.image_label = ttk.Label(right_frame, text="No image selected")
        self.image_label.grid(row=0, column=0, padx=10, pady=10)
        
        left_frame.columnconfigure(0, weight=1)
        
    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff")]
        )
        
        if file_path:
            try:
                self.original_image = Image.open(file_path).convert('L')  # Convert to grayscale
                self.display_preview()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def display_preview(self):
        if self.original_image:
            # Resize image for preview
            preview_size = (300, 300)
            self.preview_image = self.original_image.copy()
            self.preview_image.thumbnail(preview_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage for tkinter
            photo = ImageTk.PhotoImage(self.preview_image)
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo  # Keep a reference
    
    def generate_lampshade(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please select an image first")
            return
        
        try:
            # Generate cylindrical lithophane mesh
            vertices, faces = self.create_cylindrical_lithophane()
            
            # Create 3D preview
            self.show_3d_preview(vertices, faces)
            
            # Export STL
            self.export_stl(vertices, faces)
            
            messagebox.showinfo("Success", "Lampshade STL generated successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate lampshade: {str(e)}")
    
    def create_cylindrical_lithophane(self):
        # Get parameters
        height = self.height.get()
        diameter = self.diameter.get()
        radius = diameter / 2
        max_thickness = self.thickness.get()
        min_thickness = self.min_thickness.get()
        
        # Resize image to appropriate resolution
        img_width = 200  # Circumferential resolution
        img_height = int(200 * height / diameter)  # Height resolution
        
        resized_img = self.original_image.resize((img_width, img_height), Image.Resampling.LANCZOS)
        img_array = np.array(resized_img, dtype=np.float32) / 255.0
        
        # Create cylindrical coordinates
        theta = np.linspace(0, 2*np.pi, img_width, endpoint=False)
        z = np.linspace(0, height, img_height)
        
        vertices = []
        faces = []
        vertex_count = 0
        
        # Generate vertices for outer surface (lithophane)
        for i in range(img_height):
            for j in range(img_width):
                angle = theta[j]
                height_pos = z[i]
                
                # Calculate thickness based on image brightness
                brightness = img_array[i, j]
                thickness = min_thickness + (max_thickness - min_thickness) * (1 - brightness)
                outer_radius = radius + thickness
                
                # Outer surface vertex
                x_outer = outer_radius * np.cos(angle)
                y_outer = outer_radius * np.sin(angle)
                vertices.append([x_outer, y_outer, height_pos])
                
                # Inner surface vertex
                x_inner = radius * np.cos(angle)
                y_inner = radius * np.sin(angle)
                vertices.append([x_inner, y_inner, height_pos])
        
        # Generate faces
        for i in range(img_height - 1):
            for j in range(img_width):
                # Current vertex indices
                curr_outer = (i * img_width + j) * 2
                curr_inner = curr_outer + 1
                
                # Next vertex indices (wrap around for j)
                next_j = (j + 1) % img_width
                next_outer = (i * img_width + next_j) * 2
                next_inner = next_outer + 1
                
                # Next row vertex indices
                next_row_outer = ((i + 1) * img_width + j) * 2
                next_row_inner = next_row_outer + 1
                next_row_next_outer = ((i + 1) * img_width + next_j) * 2
                next_row_next_inner = next_row_next_outer + 1
                
                # Outer surface faces
                faces.append([curr_outer, next_outer, next_row_outer])
                faces.append([next_outer, next_row_next_outer, next_row_outer])
                
                # Inner surface faces
                faces.append([curr_inner, next_row_inner, next_inner])
                faces.append([next_inner, next_row_inner, next_row_next_inner])
                
                # Side faces (connecting outer and inner)
                if i == 0:  # Bottom edge
                    faces.append([curr_outer, curr_inner, next_inner])
                    faces.append([curr_outer, next_inner, next_outer])
                
                if i == img_height - 2:  # Top edge
                    faces.append([next_row_outer, next_row_next_outer, next_row_next_inner])
                    faces.append([next_row_outer, next_row_next_inner, next_row_inner])
        
        return np.array(vertices), np.array(faces)
    
    def show_3d_preview(self, vertices, faces):
        # Create 3D matplotlib figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the mesh
        for face in faces[:1000]:  # Limit faces for performance
            if len(face) == 3 and all(i < len(vertices) for i in face):
                triangle = vertices[face]
                ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2], 
                               alpha=0.6, shade=True)
        
        # Set labels and title
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('Lithophane Lampshade Preview')
        
        # Set equal aspect ratio
        max_range = max(np.max(vertices) - np.min(vertices))
        ax.set_xlim([-max_range/2, max_range/2])
        ax.set_ylim([-max_range/2, max_range/2])
        ax.set_zlim([0, max_range])
        
        plt.tight_layout()
        plt.show()
    
    def export_stl(self, vertices, faces):
        # Create STL mesh
        lampshade_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        
        for i, face in enumerate(faces):
            for j in range(3):
                if face[j] < len(vertices):
                    lampshade_mesh.vectors[i][j] = vertices[face[j]]
        
        # Save STL file
        output_path = filedialog.asksaveasfilename(
            title="Save STL file",
            defaultextension=".stl",
            filetypes=[("STL files", "*.stl")]
        )
        
        if output_path:
            lampshade_mesh.save(output_path)
            print(f"STL file saved to: {output_path}")

def main():
    root = tk.Tk()
    app = LithophaneLampshadeGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
