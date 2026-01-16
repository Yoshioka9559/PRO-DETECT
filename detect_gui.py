import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import numpy as np

class ProductDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Betty Crocker & Haagen-Dazs Product Detector")
        self.root.geometry("1200x850")
        self.root.configure(bg="#1a1a2e")
        
        # Load the model
        self.model_path = 'runs/detect/betty_haagen/weights/best.pt'
        self.confidence = 0.25
        self.model = None
        self.current_image = None
        self.current_image_path = None
        self.logo_image = None
        self.webcam = None
        self.webcam_running = False
        
        self.load_logo()
        self.setup_ui()
        self.load_model()
        
    def load_logo(self):
        """Load the General Mills logo"""
        try:
            logo_path = r"GMI Image\Image result for general mills logo.png"
            logo = Image.open(logo_path)
            # Resize logo to fit in header (bigger size)
            logo.thumbnail((400, 100), Image.Resampling.LANCZOS)
            self.logo_image = ImageTk.PhotoImage(logo)
        except Exception as e:
            print(f"Could not load logo: {e}")
            self.logo_image = None
        
    def setup_ui(self):
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Header Frame
        header_frame = tk.Frame(self.root, bg="white", height=120)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        # Container for logo and title
        header_container = tk.Frame(header_frame, bg="white")
        header_container.pack(expand=True)
        
        # Logo
        if self.logo_image:
            logo_label = tk.Label(header_container, image=self.logo_image, bg="white")
            logo_label.pack(side=tk.LEFT, padx=(0, 15))
        
        title_label = tk.Label(header_container, text="ProDetect", 
                              font=("Segoe UI", 26, "bold"), 
                              bg="white", fg="#0A4A8E")
        title_label.pack(side=tk.LEFT)
        
        # Control Panel Frame
        control_frame = tk.Frame(self.root, bg="white", pady=15)
        control_frame.pack(fill=tk.X)
        
        # Button container
        button_container = tk.Frame(control_frame, bg="white")
        button_container.pack()
        
        # Styled buttons with hover effect
        btn_style = {
            "font": ("Segoe UI", 12, "bold"),
            "relief": tk.RAISED,
            "bd": 4,
            "cursor": "hand2",
            "padx": 30,
            "pady": 12
        }
        
        self.upload_btn = tk.Button(button_container, text="📁 Upload Image", 
                                    command=self.upload_image,
                                    bg="#00d9ff", fg="white", 
                                    activebackground="#00b8e6",
                                    **btn_style)
        self.upload_btn.grid(row=0, column=0, padx=10)
        
        self.webcam_btn = tk.Button(button_container, text="📹 Webcam", 
                                    command=self.toggle_webcam,
                                    bg="#9C27B0", fg="white",
                                    activebackground="#7B1FA2",
                                    **btn_style)
        self.webcam_btn.grid(row=0, column=1, padx=10)
        
        self.detect_btn = tk.Button(button_container, text="🔍 Detect Products", 
                                    command=self.detect_products,
                                    bg="#4CAF50", fg="white",
                                    activebackground="#45a049",
                                    state=tk.DISABLED, **btn_style)
        self.detect_btn.grid(row=0, column=2, padx=10)
        
        self.clear_btn = tk.Button(button_container, text="🗑️ Clear", 
                                   command=self.clear_image,
                                   bg="#e94560", fg="white",
                                   activebackground="#d63851",
                                   **btn_style)
        self.clear_btn.grid(row=0, column=3, padx=10)
        
        # Confidence slider with modern styling
        slider_frame = tk.Frame(control_frame, bg="white")
        slider_frame.pack(pady=10)
        
        tk.Label(slider_frame, text="Confidence Threshold:", 
                font=("Segoe UI", 11), bg="white", fg="#0A4A8E").grid(row=0, column=0, padx=10)
        
        self.conf_slider = tk.Scale(slider_frame, from_=0.1, to=1.0, 
                                   resolution=0.05, orient=tk.HORIZONTAL,
                                   length=250, command=self.update_confidence,
                                   bg="white", fg="#0A4A8E", troughcolor="#E0E0E0",
                                   highlightthickness=0, font=("Segoe UI", 9))
        self.conf_slider.set(self.confidence)
        self.conf_slider.grid(row=0, column=1, padx=5)
        
        self.conf_label = tk.Label(slider_frame, text=f"{self.confidence:.2f}",
                                   font=("Segoe UI", 12, "bold"), 
                                   bg="white", fg="#0A4A8E", width=5)
        self.conf_label.grid(row=0, column=2, padx=10)
        
        # Total count display with gradient effect
        count_frame = tk.Frame(self.root, bg="#0A4A8E", height=70, relief=tk.RIDGE, bd=5)
        count_frame.pack(fill=tk.X, pady=5)
        count_frame.pack_propagate(False)
        
        self.count_label = tk.Label(count_frame, text="TOTAL: 0 Products",
                                    font=("Segoe UI", 32, "bold"), 
                                    fg="#FFD700", bg="#0A4A8E", relief=tk.SUNKEN, bd=3)
        self.count_label.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        
        # Image display with border
        image_container = tk.Frame(self.root, bg="#0A4A8E", padx=10, pady=10)
        image_container.pack(fill=tk.BOTH, expand=True)
        
        # Add scrollbars
        h_scrollbar = tk.Scrollbar(image_container, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        v_scrollbar = tk.Scrollbar(image_container, orient=tk.VERTICAL)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas = tk.Canvas(image_container, bg="#084080", 
                               width=1100, height=450,
                               highlightthickness=3, highlightbackground="#FFD700",
                               relief=tk.SUNKEN, bd=3,
                               xscrollcommand=h_scrollbar.set,
                               yscrollcommand=v_scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        h_scrollbar.config(command=self.canvas.xview)
        v_scrollbar.config(command=self.canvas.yview)
        
        # Results frame with styled text area
        results_container = tk.Frame(self.root, bg="#0A4A8E", padx=20, pady=10)
        results_container.pack(fill=tk.X)
        
        results_label = tk.Label(results_container, text="📊 Detection Results",
                                font=("Segoe UI", 12, "bold"), 
                                bg="#0A4A8E", fg="white", anchor=tk.W)
        results_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.results_text = tk.Text(results_container, height=5, width=100,
                                   font=("Consolas", 10), bg="#084080", 
                                   fg="white", relief=tk.RIDGE,
                                   insertbackground="white", bd=3)
        self.results_text.pack(fill=tk.X)
        
        # Status bar
        self.status_label = tk.Label(self.root, text="✓ Ready. Please upload an image.",
                                    font=("Segoe UI", 10), anchor=tk.W, 
                                    relief=tk.FLAT, bg="#0A4A8E", fg="white", 
                                    padx=10, pady=5)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
        
    def load_model(self):
        try:
            self.status_label.config(text="⏳ Loading model...", fg="#ffa500")
            self.root.update()
            self.model = YOLO(self.model_path)
            self.status_label.config(text=f"✓ Model loaded: {self.model_path}", fg="#4CAF50")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.status_label.config(text="❌ Error loading model", fg="#e94560")
            
    def update_confidence(self, value):
        self.confidence = float(value)
        self.conf_label.config(text=f"{self.confidence:.2f}")
        
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                      ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.current_image_path = file_path
                self.current_image = cv2.imread(file_path)
                
                if self.current_image is None:
                    messagebox.showerror("Error", "Could not read the image file")
                    return
                
                # Display the image
                self.display_image(self.current_image)
                self.detect_btn.config(state=tk.NORMAL)
                self.status_label.config(text=f"✓ Image loaded: {file_path}", fg="#4CAF50")
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "📷 Image uploaded successfully. Click 'Detect Products' to analyze.\n")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                
    def display_image(self, cv_image):
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Resize to fit canvas while maintaining aspect ratio
        h, w = rgb_image.shape[:2]
        canvas_w = 1100
        canvas_h = 450
        
        scale = min(canvas_w/w, canvas_h/h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(rgb_image, (new_w, new_h))
        
        # Convert to PIL Image
        pil_image = Image.fromarray(resized)
        self.photo = ImageTk.PhotoImage(pil_image)
        
        # Display on canvas centered
        self.canvas.delete("all")
        x = (canvas_w - new_w) // 2
        y = (canvas_h - new_h) // 2
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo)
        self.canvas.config(scrollregion=(0, 0, canvas_w, canvas_h))
        
    def detect_products(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please upload an image first")
            return
            
        try:
            self.status_label.config(text="🔍 Detecting products...", fg="#ffa500")
            self.root.update()
            
            # Run detection
            results = self.model(self.current_image, conf=self.confidence)
            result = results[0]
            
            # Count detections
            class_counts = {}
            total_detections = 0
            detection_details = []
            
            if len(result.boxes) > 0:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    class_name = self.model.names[cls_id]
                    confidence = float(box.conf[0])
                    
                    if class_name not in class_counts:
                        class_counts[class_name] = 0
                    class_counts[class_name] += 1
                    total_detections += 1
                    
                    detection_details.append(f"{class_name}: {confidence:.1%}")
            
            # Display annotated image
            annotated_image = result.plot()
            self.display_image(annotated_image)
            
            # Update the count label above the image
            self.count_label.config(text=f"TOTAL: {total_detections} Products")
            
            # Update results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "="*90 + "\n")
            self.results_text.insert(tk.END, f"   DETECTION RESULTS (Confidence Threshold: {self.confidence:.2f})\n")
            self.results_text.insert(tk.END, "="*90 + "\n\n")
            
            if total_detections > 0:
                self.results_text.insert(tk.END, f"📦 TOTAL PRODUCTS DETECTED: {total_detections}\n\n")
                self.results_text.insert(tk.END, "📊 Product Breakdown:\n")
                
                for class_name, count in class_counts.items():
                    self.results_text.insert(tk.END, f"   • {class_name}: {count} unit(s)\n")
                
                self.results_text.insert(tk.END, f"\n🔍 Detection Details:\n")
                for detail in detection_details:
                    self.results_text.insert(tk.END, f"   - {detail}\n")
                    
                self.status_label.config(text=f"✓ Detection complete! Found {total_detections} product(s)", fg="#4CAF50")
            else:
                self.results_text.insert(tk.END, "⚠️ No products detected!\n\n")
                self.results_text.insert(tk.END, "💡 Possible reasons:\n")
                self.results_text.insert(tk.END, "   • Product not visible or too small\n")
                self.results_text.insert(tk.END, "   • Poor lighting or image quality\n")
                self.results_text.insert(tk.END, "   • Confidence threshold too high\n")
                self.results_text.insert(tk.END, "   • Object differs from training data\n")
                self.status_label.config(text="⚠️ No products detected", fg="#ffa500")
                
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {str(e)}")
            self.status_label.config(text="❌ Detection failed", fg="#e94560")
            
    def clear_image(self):
        self.stop_webcam()
        self.current_image = None
        self.current_image_path = None
        self.canvas.delete("all")
        self.results_text.delete(1.0, tk.END)
        self.count_label.config(text="TOTAL: 0 Products")
        self.detect_btn.config(state=tk.DISABLED)
        self.status_label.config(text="✓ Ready. Please upload an image.", fg="white")
    
    def toggle_webcam(self):
        """Start or stop webcam detection"""
        if self.webcam_running:
            self.stop_webcam()
        else:
            # Ask user to choose between webcam or video file
            choice = messagebox.askquestion("Webcam or Video", 
                                           "Do you want to use live webcam?\n\nClick 'Yes' for webcam or 'No' to select a video file.")
            if choice == 'yes':
                self.start_webcam()
            else:
                self.start_video_file()
    
    def start_webcam(self):
        """Start webcam detection"""
        try:
            self.webcam = cv2.VideoCapture(0)
            if not self.webcam.isOpened():
                messagebox.showerror("Error", "Could not open webcam")
                return
            
            self.webcam_running = True
            self.webcam_btn.config(text="⏹️ Stop", bg="#f44336")
            self.upload_btn.config(state=tk.DISABLED)
            self.detect_btn.config(state=tk.DISABLED)
            self.status_label.config(text="📹 Webcam running - Detecting in real-time...", fg="#4CAF50")
            self.update_webcam()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start webcam: {str(e)}")
            self.stop_webcam()
    
    def start_video_file(self):
        """Start video file detection"""
        file_path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.flv"),
                      ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.webcam = cv2.VideoCapture(file_path)
                if not self.webcam.isOpened():
                    messagebox.showerror("Error", "Could not open video file")
                    return
                
                self.webcam_running = True
                self.webcam_btn.config(text="⏹️ Stop Video", bg="#f44336")
                self.upload_btn.config(state=tk.DISABLED)
                self.detect_btn.config(state=tk.DISABLED)
                self.status_label.config(text=f"🎬 Playing video: {file_path}", fg="#4CAF50")
                self.update_webcam()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open video: {str(e)}")
                self.stop_webcam()
    
    def stop_webcam(self):
        """Stop webcam detection"""
        self.webcam_running = False
        if self.webcam:
            self.webcam.release()
            self.webcam = None
        self.webcam_btn.config(text="📹 Webcam", bg="#9C27B0")
        self.upload_btn.config(state=tk.NORMAL)
        self.status_label.config(text="✓ Webcam stopped.", fg="white")
    
    def update_webcam(self):
        """Update webcam frame and run detection"""
        if not self.webcam_running:
            return
        
        ret, frame = self.webcam.read()
        if ret:
            # Run detection on frame
            results = self.model(frame, conf=self.confidence)
            result = results[0]
            
            # Count detections
            total_detections = len(result.boxes)
            class_counts = {}
            
            if len(result.boxes) > 0:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    class_name = self.model.names[cls_id]
                    if class_name not in class_counts:
                        class_counts[class_name] = 0
                    class_counts[class_name] += 1
            
            # Update count label
            self.count_label.config(text=f"TOTAL: {total_detections} Products")
            
            # Display annotated frame
            annotated_frame = result.plot()
            self.display_image(annotated_frame)
            
            # Update results text
            self.results_text.delete(1.0, tk.END)
            if total_detections > 0:
                self.results_text.insert(tk.END, f"📦 Live Detection: {total_detections} product(s)\n")
                for class_name, count in class_counts.items():
                    self.results_text.insert(tk.END, f"   • {class_name}: {count}\n")
            else:
                self.results_text.insert(tk.END, "No products detected in frame\n")
        
        # Schedule next update (30 FPS)
        if self.webcam_running:
            self.root.after(33, self.update_webcam)


if __name__ == "__main__":
    root = tk.Tk()
    app = ProductDetectorGUI(root)
    root.mainloop()
