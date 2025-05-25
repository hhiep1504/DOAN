import os
import cv2
import numpy as np
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import threading
from pathlib import Path
from ultralytics import YOLO

class PersonAttributeLabelingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Person Attribute Labeling Tool")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        
        # Variables
        self.frames_dir = tk.StringVar()
        self.model_path = tk.StringVar(value="yolov10n.pt")
        self.threshold = tk.DoubleVar(value=0.5)
        self.current_frame_idx = 0
        self.frames_list = []
        self.frame_data = {}
        self.person_attributes = {}
        self.current_person_id = None
        self.tracking_in_progress = False
        self.model = None
        self.original_frame = None
        self.displayed_frame = None
        self.scale_factor = 1.0
        
        # Attribute options
        self.attribute_options = {
            "gender": ["Male", "Female", "Không xác định"],
            "age": ["0-17", "18-54", "55+", "Unknown"],
            "ethnicity": ["White", "Black", "Asian","Indian", "Unknown"],
            "beard": ["Yes", "No", "Unknown"],
            "glasses": ["Normal glass", "Sun glass", "No", "Unknown"],
            "accessories": ["Bag", "Backpack Bag", "Rolling Bag", "Umbrella", "Sport Bag", "Market Bag", "Nothing", "Unknown"]
        }
        
        # Create attributes variables
        self.attribute_vars = {
            "gender": tk.StringVar(value="0"),
            "age": tk.StringVar(value="0"),
            "ethnicity": tk.StringVar(value="0"),
            "beard": tk.StringVar(value="0"),
            "glasses": tk.StringVar(value="0"),
            "accessories": tk.StringVar(value="0")
        }
        
        self.create_widgets()
    
    def create_widgets(self):
        # Create main frames
        self.left_panel = ttk.Frame(self.root)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.right_panel = ttk.Frame(self.root)
        # It's often good practice for both main panels to be expandable,
        # or for one to have a more fixed size.
        # If right_panel content is fixed width, expand=False is fine.
        # If it should also share space, add expand=True.
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10) # Added expand=True for balance, can be False if preferred

        # --- MODIFICATION START ---
        # Pack control_frame (at the bottom of left_panel) BEFORE video_frame
        # This ensures control_frame gets its necessary height, and video_frame takes the rest.

        self.control_frame = ttk.LabelFrame(self.left_panel, text="Controls")
        # Pack it at the bottom of left_panel. It will take its natural height.
        self.control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5, ipady=2) # ipady adds some internal vertical padding

        # Video display and controls
        self.video_frame = ttk.LabelFrame(self.left_panel, text="Video Preview")
        # This will now fill the remaining space in left_panel above control_frame.
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        # --- MODIFICATION END ---

        self.canvas = tk.Canvas(self.video_frame, bg="black", width=800, height=600) # Initial preferred size
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5) # Canvas expands within video_frame

        # Control frame's children (these are packed into self.control_frame, which is already packed)
        # File selection
        self.dir_frame = ttk.Frame(self.control_frame)
        self.dir_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(self.dir_frame, text="Frames Directory:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(self.dir_frame, textvariable=self.frames_dir, width=40).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(self.dir_frame, text="Browse...", command=self.browse_frames_dir).pack(side=tk.LEFT, padx=5)

        # Model selection
        self.model_frame = ttk.Frame(self.control_frame)
        self.model_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(self.model_frame, text="YOLO Model Path:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(self.model_frame, textvariable=self.model_path, width=40).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(self.model_frame, text="Browse...", command=self.browse_model_path).pack(side=tk.LEFT, padx=5)

        # Thêm threshold input
        self.threshold_frame = ttk.Frame(self.control_frame)
        self.threshold_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(self.threshold_frame, text="YOLO Threshold:").pack(side=tk.LEFT, padx=5)
        threshold_entry = ttk.Entry(self.threshold_frame, textvariable=self.threshold, width=10)
        threshold_entry.pack(side=tk.LEFT, padx=5)

        # Action buttons
        self.action_frame = ttk.Frame(self.control_frame)
        self.action_frame.pack(fill=tk.X, padx=5, pady=5)

        # Tạo frame con bên trái cho các nút điều khiển chính
        left_action_frame = ttk.Frame(self.action_frame)
        left_action_frame.pack(side=tk.LEFT)

        self.track_button = ttk.Button(left_action_frame, text="Start Tracking", command=self.start_tracking)
        self.track_button.pack(side=tk.LEFT, padx=5)

        self.save_button = ttk.Button(left_action_frame, text="Save Data", command=self.save_data, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=5)

        self.load_button = ttk.Button(left_action_frame, text="Load Data", command=self.load_data)
        self.load_button.pack(side=tk.LEFT, padx=5)
        
        self.merge_button = ttk.Button(left_action_frame, text="Merge IDs", command=self.merge_ids)
        self.merge_button.pack(side=tk.LEFT, padx=5)
        
        self.delete_button = ttk.Button(left_action_frame, text="Delete ID", command=self.delete_id)
        self.delete_button.pack(side=tk.LEFT, padx=5)
        
        self.split_button = ttk.Button(left_action_frame, text="Split ID", command=self.split_id)
        self.split_button.pack(side=tk.LEFT, padx=5)

        # Tạo frame con bên phải cho các nút điều hướng frame
        right_action_frame = ttk.Frame(self.action_frame)
        right_action_frame.pack(side=tk.RIGHT)

        self.prev_frame_button = ttk.Button(right_action_frame, text="Previous Frame", command=self.show_prev_frame, state=tk.DISABLED)
        self.prev_frame_button.pack(side=tk.LEFT, padx=5)

        self.frame_counter_label = ttk.Label(right_action_frame, text="Frame: 0/0")
        self.frame_counter_label.pack(side=tk.LEFT, padx=5)

        self.next_frame_button = ttk.Button(right_action_frame, text="Next Frame", command=self.show_next_frame, state=tk.DISABLED)
        self.next_frame_button.pack(side=tk.LEFT, padx=5)

        # Right panel - Person attributes
        self.person_list_frame = ttk.LabelFrame(self.right_panel, text="Detected Persons")
        self.person_list_frame.pack(fill=tk.BOTH, padx=5, pady=5) # No expand here, it's fine, its parent right_panel expands

        # Person list with scrollbar
        self.person_listbox_frame = ttk.Frame(self.person_list_frame)
        self.person_listbox_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.person_listbox = tk.Listbox(self.person_listbox_frame, height=10)
        self.person_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.person_listbox.bind('<<ListboxSelect>>', self.on_person_selected)

        person_scrollbar = ttk.Scrollbar(self.person_listbox_frame, orient=tk.VERTICAL, command=self.person_listbox.yview)
        person_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.person_listbox.config(yscrollcommand=person_scrollbar.set)

        # Person preview image
        self.person_preview_frame = ttk.LabelFrame(self.right_panel, text="Person Preview")
        self.person_preview_frame.pack(fill=tk.BOTH, padx=5, pady=5, ipady=5) # No expand here is fine

        self.person_canvas = tk.Canvas(self.person_preview_frame, bg="black", height=150) # Fixed height
        self.person_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5) # Expands within its parent

        # Attributes frame with a summary display
        self.attributes_frame = ttk.LabelFrame(self.right_panel, text="Person Attributes")
        self.attributes_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5) # This one expands vertically in right_panel

        # Create a summary frame for showing current attributes
        self.attr_summary_frame = ttk.Frame(self.attributes_frame)
        self.attr_summary_frame.pack(fill=tk.X, padx=5, pady=5)

        self.attr_summary_text = tk.Text(self.attr_summary_frame, height=6, width=40, state=tk.DISABLED)
        self.attr_summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        summary_scrollbar = ttk.Scrollbar(self.attr_summary_frame, orient=tk.VERTICAL, command=self.attr_summary_text.yview)
        summary_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.attr_summary_text.config(yscrollcommand=summary_scrollbar.set)

        # Create a frame for attribute inputs
        self.attr_input_frame = ttk.Frame(self.attributes_frame)
        self.attr_input_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create a canvas with scrollbar for attributes
        self.attr_canvas = tk.Canvas(self.attr_input_frame)
        self.attr_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        attr_scrollbar = ttk.Scrollbar(self.attr_input_frame, orient=tk.VERTICAL, command=self.attr_canvas.yview)
        attr_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.attr_canvas.configure(yscrollcommand=attr_scrollbar.set)
        self.attr_canvas.bind('<Configure>', lambda e: self.attr_canvas.configure(scrollregion=self.attr_canvas.bbox("all")))

        # Create a frame inside the canvas for the attributes
        self.attr_inner_frame = ttk.Frame(self.attr_canvas)
        self.attr_canvas.create_window((0, 0), window=self.attr_inner_frame, anchor="nw")

        # Create attribute sections with input fields
        self.create_attribute_widgets()

        # Create a legend frame
        self.legend_frame = ttk.LabelFrame(self.right_panel, text="Legend")
        self.legend_frame.pack(fill=tk.BOTH, padx=5, pady=5) # No expand here, fine

        # Create a notebook for attribute legends
        self.legend_notebook = ttk.Notebook(self.legend_frame)
        self.legend_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create a tab for each attribute's legend
        for attr_name, options in self.attribute_options.items():
            legend_tab = ttk.Frame(self.legend_notebook)
            self.legend_notebook.add(legend_tab, text=attr_name.capitalize())

            legend_text = tk.Text(legend_tab, height=10, width=40) # Fixed height
            legend_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            # Add scrollbar to legend text
            legend_scrollbar = ttk.Scrollbar(legend_text, orient=tk.VERTICAL, command=legend_text.yview)
            legend_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            legend_text.config(yscrollcommand=legend_scrollbar.set)

            # Populate legend
            for i, option in enumerate(options):
                legend_text.insert(tk.END, f"{i}: {option}\n")

            legend_text.config(state=tk.DISABLED)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Bind right click on canvas
        self.canvas.bind("<Button-3>", self.show_context_menu)

        # Thêm binding cho phím tắt
        self.root.bind('<Left>', lambda event: self.show_prev_frame())
        self.root.bind('<Right>', lambda event: self.show_next_frame())

        
    def create_attribute_widgets(self):
        # Create a compact view for all attributes
        attr_main_frame = ttk.Frame(self.attr_inner_frame)
        attr_main_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create a grid for all attribute inputs
        row = 0
        for attr_name in self.attribute_options.keys():
            # Add label and entry for numerical input
            ttk.Label(attr_main_frame, text=f"{attr_name.capitalize()}:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            
            # Create and configure spinbox for input
            spinbox = ttk.Spinbox(
                attr_main_frame,
                from_=0,
                to=8,
                width=5,
                textvariable=self.attribute_vars[attr_name],
                command=self.update_attributes
            )
            spinbox.grid(row=row, column=1, padx=5, pady=2)
            
            # Bind validation
            spinbox.bind('<FocusOut>', lambda e, attr=attr_name: self.validate_attribute(attr))
            spinbox.bind('<Return>', lambda e, attr=attr_name: self.validate_attribute(attr))
            
            # Add a preview of the current value
            value_preview = ttk.Label(attr_main_frame, text=self.attribute_options[attr_name][0])
            value_preview.grid(row=row, column=2, sticky=tk.W, padx=5, pady=2)
            
            # Store reference to update later
            self.attribute_vars[attr_name].trace_add("write", lambda *args, attr=attr_name, label=value_preview: 
                            self.update_value_preview(attr, label))
            
            row += 1
        
        # Add a button to update all attributes at once
        update_btn = ttk.Button(attr_main_frame, text="Update All Attributes", command=self.update_attributes)
        update_btn.grid(row=row, column=0, columnspan=3, padx=5, pady=10)
    
    def update_value_preview(self, attr_name, label):
        try:
            value = int(self.attribute_vars[attr_name].get())
            if 0 <= value <= 8:
                label.config(text=self.attribute_options[attr_name][value])
            else:
                label.config(text="Invalid value")
        except ValueError:
            label.config(text="Invalid value")
    
    def validate_attribute(self, attr_name):
        """Validate the attribute value is between 0-8"""
        try:
            value = int(self.attribute_vars[attr_name].get())
            if value < 0 or value > 8:
                self.attribute_vars[attr_name].set("0")
        except ValueError:
            self.attribute_vars[attr_name].set("0")
        
        self.update_attributes()
    
    def browse_frames_dir(self):
        directory = filedialog.askdirectory(title="Select Frames Directory")
        if directory:
            self.frames_dir.set(directory)
    
    def browse_model_path(self):
        file_path = filedialog.askopenfilename(
            title="Select YOLO Model File",
            filetypes=[("YOLO Model", "*.pt"), ("All Files", "*.*")]
        )
        if file_path:
            self.model_path.set(file_path)
    
    def start_tracking(self):
        if not self.frames_dir.get():
            messagebox.showerror("Error", "Please select a frames directory.")
            return
        
        if not os.path.exists(self.frames_dir.get()):
            messagebox.showerror("Error", "Selected frames directory does not exist.")
            return
        
        # Start tracking in a separate thread to keep UI responsive
        self.tracking_in_progress = True
        self.track_button.config(state=tk.DISABLED)
        self.status_var.set("Loading model and tracking objects...")
        
        tracking_thread = threading.Thread(target=self.tracking_process)
        tracking_thread.daemon = True
        tracking_thread.start()
    
    def tracking_process(self):
        try:
            # Load the model
            self.model = YOLO(self.model_path.get())
            
            # Get the frames
            self.frames_list = sorted([
                os.path.join(self.frames_dir.get(), f) 
                for f in os.listdir(self.frames_dir.get()) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            
            if not self.frames_list:
                self.root.after(0, lambda: messagebox.showerror("Error", "No image frames found in the selected directory."))
                self.tracking_in_progress = False
                self.root.after(0, lambda: self.track_button.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.status_var.set("Ready"))
                return
            
            # Reset data
            self.frame_data = {}
            self.person_attributes = {}
            self.current_frame_idx = 0
            
            # Process each frame
            for frame_idx, frame_path in enumerate(self.frames_list):
                frame = cv2.imread(frame_path)
                if frame is None:
                    continue
                
                # Update status
                self.root.after(0, lambda idx=frame_idx: self.status_var.set(f"Processing frame {idx+1}/{len(self.frames_list)}..."))
                
                # Track objects in the frame with threshold
                results = self.model.track(frame, persist=True, classes=0, tracker="bytetrack.yaml", conf=self.threshold.get())
                
                # Store frame data
                self.frame_data[frame_idx] = []
                
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().numpy()
                    
                    for i, box_coords in enumerate(boxes):
                        track_id = int(track_ids[i])
                        x1, y1, x2, y2 = map(float, box_coords)
                        
                        # Initialize person attributes if not exists
                        if track_id not in self.person_attributes:
                            self.person_attributes[track_id] = {
                                "gender": 0, "age": 0, "ethnicity": 0,
                                "beard": 0, "glasses": 0, "accessories": 0
                            }
                        
                        # Store bounding box info for this frame
                        self.frame_data[frame_idx].append({
                            "track_id": track_id,
                            "bbox": [x1, y1, x2, y2],
                            "attributes": self.person_attributes[track_id]
                        })
            
            # Update UI after tracking is done
            self.root.after(0, self.update_ui_after_tracking)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"An error occurred during tracking: {str(e)}"))
            self.tracking_in_progress = False
            self.root.after(0, lambda: self.track_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.status_var.set("Ready"))
    
    def update_ui_after_tracking(self):
        self.tracking_in_progress = False
        self.track_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.NORMAL)
        self.prev_frame_button.config(state=tk.NORMAL)
        self.next_frame_button.config(state=tk.NORMAL)
        
        # Update person listbox
        self.update_person_list()
        
        # Show the first frame
        self.show_frame(0)
        
        self.status_var.set(f"Tracking completed. Found {len(self.person_attributes)} persons in {len(self.frames_list)} frames.")
    
    def update_person_list(self):
        self.person_listbox.delete(0, tk.END)
        for person_id in sorted(self.person_attributes.keys()):
            self.person_listbox.insert(tk.END, f"Person ID: {person_id}")
    
    def show_frame(self, frame_idx):
        if not self.frames_list or frame_idx < 0 or frame_idx >= len(self.frames_list):
            return
        
        self.current_frame_idx = frame_idx
        frame_path = self.frames_list[frame_idx]
        
        # Read and display the frame
        frame = cv2.imread(frame_path)
        if frame is None:
            self.status_var.set(f"Error: Could not read frame {frame_path}")
            return
        
        self.original_frame = frame.copy()
        
        # Draw bounding boxes for this frame
        if frame_idx in self.frame_data:
            for person_data in self.frame_data[frame_idx]:
                track_id = person_data["track_id"]
                bbox = person_data["bbox"]
                x1, y1, x2, y2 = map(int, bbox)
                
                # Draw rectangle and ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Convert to RGB for display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.displayed_frame = rgb_frame.copy()
        
        # Thay đổi ở đây: Lấy kích thước cố định của canvas
        canvas_width = 800  # Kích thước cố định
        canvas_height = 600  # Kích thước cố định
        
        # Calculate aspect ratio
        img_height, img_width = rgb_frame.shape[:2]
        ratio = min(canvas_width/img_width, canvas_height/img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        
        # Resize image
        resized_img = cv2.resize(rgb_frame, (new_width, new_height))
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(image=Image.fromarray(resized_img))
        
        # Update canvas - Không thay đổi kích thước canvas nữa
        self.canvas.delete("all")
        
        # Căn giữa ảnh trong canvas
        x_offset = (canvas_width - new_width) // 2
        y_offset = (canvas_height - new_height) // 2
        self.canvas.create_image(x_offset, y_offset, image=photo, anchor=tk.NW)
        self.canvas.image = photo  # Keep a reference
        
        # Lưu tỷ lệ scale để dùng cho việc xử lý click
        self.scale_factor = ratio
        
        # Update frame counter
        self.frame_counter_label.config(text=f"Frame: {frame_idx+1}/{len(self.frames_list)}")
    
    def show_prev_frame(self):
        if self.current_frame_idx > 0:
            self.show_frame(self.current_frame_idx - 1)
    
    def show_next_frame(self):
        if self.current_frame_idx < len(self.frames_list) - 1:
            self.show_frame(self.current_frame_idx + 1)
    
    def on_person_selected(self, event):
        selection = self.person_listbox.curselection()
        if not selection:
            return
        
        # Get the selected person ID
        person_item = self.person_listbox.get(selection[0])
        try:
            self.current_person_id = int(person_item.split(":")[1].strip())
        except (ValueError, IndexError):
            return
        
        # Update attribute spinboxes
        self.update_attribute_spinboxes()
        
        # Update attribute summary
        self.update_attribute_summary()
        
        # Show person image from a frame with this ID
        self.show_person_preview()
    
    def update_attribute_spinboxes(self):
        if self.current_person_id is None or self.current_person_id not in self.person_attributes:
            return
        
        # Set spinbox values according to stored attributes
        attrs = self.person_attributes[self.current_person_id]
        for attr_name, value in attrs.items():
            if attr_name in self.attribute_vars:
                self.attribute_vars[attr_name].set(str(value))
    
    def update_attribute_summary(self):
        """Update the summary text widget with all current attribute values"""
        if self.current_person_id is None or self.current_person_id not in self.person_attributes:
            return
        
        # Enable text widget for editing
        self.attr_summary_text.config(state=tk.NORMAL)
        
        # Clear current content
        self.attr_summary_text.delete(1.0, tk.END)
        
        # Add title
        self.attr_summary_text.insert(tk.END, f"Attributes for Person ID: {self.current_person_id}\n\n")
        
        # Add each attribute with its text value
        attrs = self.person_attributes[self.current_person_id]
        for attr_name, value in attrs.items():
            text_value = self.attribute_options[attr_name][value]
            self.attr_summary_text.insert(tk.END, f"{attr_name.capitalize()}: {value} - {text_value}\n")
        
        # Disable text widget to prevent editing
        self.attr_summary_text.config(state=tk.DISABLED)
    
    def show_person_preview(self):
        if self.current_person_id is None:
            return
        
        # Find a good frame for this person (with largest bounding box area)
        best_frame_idx = None
        best_bbox = None
        max_area = 0
        
        for frame_idx, persons_in_frame in self.frame_data.items():
            for person_data in persons_in_frame:
                if person_data["track_id"] == self.current_person_id:
                    bbox = person_data["bbox"]
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if area > max_area:
                        max_area = area
                        best_frame_idx = frame_idx
                        best_bbox = bbox
        
        if best_frame_idx is not None and best_bbox is not None:
            frame_path = self.frames_list[best_frame_idx]
            frame = cv2.imread(frame_path)
            if frame is not None:
                x1, y1, x2, y2 = map(int, best_bbox)
                
                # Ensure coordinates are within bounds
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x1 < x2 and y1 < y2:
                    person_img = frame[y1:y2, x1:x2]
                    rgb_person = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
                    
                    # Resize to fit the canvas
                    canvas_width = self.person_canvas.winfo_width()
                    canvas_height = self.person_canvas.winfo_height()
                    
                    if canvas_width > 1 and canvas_height > 1:
                        img_h, img_w = rgb_person.shape[:2]
                        ratio = min(canvas_width/img_w, canvas_height/img_h)
                        new_w = int(img_w * ratio)
                        new_h = int(img_h * ratio)
                        
                        resized_person = cv2.resize(rgb_person, (new_w, new_h))
                        photo = ImageTk.PhotoImage(image=Image.fromarray(resized_person))
                        
                        self.person_canvas.delete("all")
                        self.person_canvas.create_image(
                            (canvas_width - new_w) // 2,
                            (canvas_height - new_h) // 2,
                            image=photo, anchor=tk.NW
                        )
                        self.person_canvas.image = photo  # Keep a reference
    
    def update_attributes(self, event=None):
        if self.current_person_id is not None:
            # Update attributes in the dictionary
            for attr_name, var in self.attribute_vars.items():
                try:
                    value = int(var.get())
                    if 0 <= value <= 8:  # Ensure value is in valid range
                        self.person_attributes[self.current_person_id][attr_name] = value
                except ValueError:
                    # If value is not a valid integer, reset to 0
                    self.person_attributes[self.current_person_id][attr_name] = 0
                    var.set("0")
            
            # Update all frame data for this person
            for frame_idx in self.frame_data:
                for person_data in self.frame_data[frame_idx]:
                    if person_data["track_id"] == self.current_person_id:
                        person_data["attributes"] = self.person_attributes[self.current_person_id].copy()
            
            # Update the attribute summary
            self.update_attribute_summary()
    
    def save_data(self):
        if not self.person_attributes:
            messagebox.showinfo("Info", "Không có dữ liệu để lưu.")
            return
        
        try:
            # Lưu thuộc tính của người
            attr_file = filedialog.asksaveasfilename(
                title="Lưu thuộc tính của người",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialfile="person_attributes.json"
            )
            
            if attr_file:
                with open(attr_file, "w") as f:
                    json.dump(self.person_attributes, f, indent=4)
            
            # Lưu annotations
            ann_file = filedialog.asksaveasfilename(
                title="Lưu Annotations",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialfile="annotations.json"
            )
            
            if ann_file:
                with open(ann_file, "w") as f:
                    json.dump(self.frame_data, f, indent=4)
            
            # Hỏi thư mục để lưu ảnh bbox
            if attr_file or ann_file:
                bbox_dir = filedialog.askdirectory(
                    title="Chọn thư mục để lưu ảnh bounding box"
                )
                
                if bbox_dir:
                    # Tạo thư mục bbox_image nếu chưa tồn tại
                    bbox_image_dir = os.path.join(bbox_dir, "bbox_image")
                    os.makedirs(bbox_image_dir, exist_ok=True)
                    
                    # Tạo dictionary để theo dõi số frame cho mỗi person_id
                    person_frame_counts = {}
                    
                    # Duyệt qua tất cả các frame và lưu bbox image
                    self.status_var.set("Đang lưu ảnh bounding box...")
                    total_frames = len(self.frame_data)
                    saved_count = 0
                    
                    for frame_idx, persons_in_frame in self.frame_data.items():
                        # Đọc frame gốc
                        if int(frame_idx) < len(self.frames_list):
                            frame_path = self.frames_list[int(frame_idx)]
                            frame = cv2.imread(frame_path)
                            
                            if frame is not None:
                                # Duyệt qua tất cả người trong frame
                                for person_data in persons_in_frame:
                                    track_id = person_data["track_id"]
                                    bbox = person_data["bbox"]
                                    x1, y1, x2, y2 = map(int, bbox)
                                    
                                    # Đảm bảo tọa độ nằm trong giới hạn ảnh
                                    h, w = frame.shape[:2]
                                    x1, y1 = max(0, x1), max(0, y1)
                                    x2, y2 = min(w, x2), min(h, y2)
                                    
                                    if x1 < x2 and y1 < y2:
                                        # Tạo thư mục cho person_id nếu chưa tồn tại
                                        person_dir = os.path.join(bbox_image_dir, f"person_{track_id}")
                                        os.makedirs(person_dir, exist_ok=True)
                                        
                                        # Cập nhật số frame cho person_id này
                                        if track_id not in person_frame_counts:
                                            person_frame_counts[track_id] = 0
                                        person_frame_counts[track_id] += 1
                                        
                                        # Cắt ảnh và lưu với tên mới
                                        person_img = frame[y1:y2, x1:x2]
                                        output_path = os.path.join(
                                            person_dir,
                                            f"person_{track_id}_frame_{frame_idx}.png"
                                        )
                                        cv2.imwrite(output_path, person_img)
                                        saved_count += 1
                    
                    # Cập nhật trạng thái
                    if (int(frame_idx) + 1) % 10 == 0:
                        self.status_var.set(f"Đang lưu ảnh: {int(frame_idx)+1}/{total_frames} frames...")
                    
                    # Tạo thông báo chi tiết về số lượng ảnh đã lưu cho mỗi person
                    detail_msg = "Chi tiết:\n"
                    for person_id, frame_count in sorted(person_frame_counts.items()):
                        detail_msg += f"Person {person_id}: {frame_count} frames\n"
                    
                    messagebox.showinfo("Thành công", 
                                      f"Dữ liệu đã được lưu thành công.\n"
                                      f"Tổng số ảnh đã lưu: {saved_count}\n\n{detail_msg}")
                    self.status_var.set(f"Dữ liệu đã được lưu. {saved_count} ảnh bounding box đã được lưu.")
                else:
                    messagebox.showinfo("Thành công", "Dữ liệu đã được lưu thành công (không lưu ảnh bounding box).")
                    self.status_var.set("Dữ liệu đã được lưu thành công.")
        
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu dữ liệu: {str(e)}")
    
    def load_data(self):
        try:
            # Load person attributes
            attr_file = filedialog.askopenfilename(
                title="Load Person Attributes",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if attr_file:
                with open(attr_file, "r") as f:
                    loaded_attributes = json.load(f)
                    
                    # Convert string keys to integers
                    self.person_attributes = {int(k): v for k, v in loaded_attributes.items()}
            
            # Load annotations
            ann_file = filedialog.askopenfilename(
                title="Load Annotations",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if ann_file:
                with open(ann_file, "r") as f:
                    loaded_frame_data = json.load(f)
                    
                    # Convert string keys to integers
                    self.frame_data = {int(k): v for k, v in loaded_frame_data.items()}
            
            if attr_file or ann_file:
                # Update person list
                self.update_person_list()
                
                # Enable buttons
                self.save_button.config(state=tk.NORMAL)
                
                # Check if we need to load frames
                if not self.frames_list and self.frames_dir.get():
                    self.frames_list = sorted([
                        os.path.join(self.frames_dir.get(), f) 
                        for f in os.listdir(self.frames_dir.get()) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                    ])
                    
                    if self.frames_list:
                        self.prev_frame_button.config(state=tk.NORMAL)
                        self.next_frame_button.config(state=tk.NORMAL)
                        self.show_frame(0)
                
                messagebox.showinfo("Success", "Data loaded successfully.")
                self.status_var.set("Data loaded successfully.")
        
        except Exception as e:
            messagebox.showerror("Error", f"Could not load data: {str(e)}")

    def show_context_menu(self, event):
        # Chuyển đổi tọa độ click theo tỷ lệ scale
        x = event.x / self.scale_factor
        y = event.y / self.scale_factor
        
        # Tìm bbox được click
        clicked_person = None
        if self.current_frame_idx in self.frame_data:
            for person in self.frame_data[self.current_frame_idx]:
                bbox = person["bbox"]
                if (bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]):
                    clicked_person = person
                    break
        
        if clicked_person:
            menu = tk.Menu(self.root, tearoff=0)
            current_id = clicked_person["track_id"]
            
            # Thêm menu item để thay đổi ID
            menu.add_command(label=f"Thay đổi ID {current_id}", 
                            command=lambda: self.change_bbox_id(current_id, clicked_person["bbox"]))
            
            menu.post(event.x_root, event.y_root)

    def change_bbox_id(self, current_id, bbox):
        # Hiện dialog để nhập ID mới
        new_id = simpledialog.askinteger("Thay đổi ID", 
                                        f"Nhập ID mới cho bbox (ID hiện tại: {current_id}):",
                                        parent=self.root)
        
        if new_id is not None:
            # Cập nhật ID trong frame_data
            for person in self.frame_data[self.current_frame_idx]:
                if (person["track_id"] == current_id and 
                    person["bbox"] == bbox):
                    person["track_id"] = new_id
                    break
            
            # Tạo thuộc tính mới nếu ID chưa tồn tại
            if new_id not in self.person_attributes:
                self.person_attributes[new_id] = {
                    "gender": 0, "age": 0, "ethnicity": 0,
                    "beard": 0, "glasses": 0, "accessories": 0
                }
            
            # Cập nhật UI
            self.update_person_list()
            self.show_frame(self.current_frame_idx)

    def merge_ids(self):
        # Create dialog to select IDs to merge
        dialog = tk.Toplevel(self.root)
        dialog.title("Merge Person IDs")
        dialog.geometry("800x500")  # Tăng kích thước cửa sổ để chứa ảnh preview
        
        # Tạo frame chính chia làm 2 phần
        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Frame bên trái cho danh sách ID
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Frame bên phải cho preview ảnh
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tạo label và listbox trong frame trái
        label = ttk.Label(left_frame, text="Chọn các ID để gộp (giữ Ctrl để chọn nhiều):")
        label.pack(pady=5)
        
        listbox = tk.Listbox(left_frame, selectmode=tk.MULTIPLE)
        listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tạo canvas để hiển thị ảnh preview trong frame phải
        ttk.Label(right_frame, text="Preview:").pack(pady=5)
        preview_canvas = tk.Canvas(right_frame, bg="black", width=300, height=400)
        preview_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Thêm các ID vào listbox
        for person_id in sorted(self.person_attributes.keys()):
            listbox.insert(tk.END, str(person_id))
        
        def on_select(event):
            # Xử lý khi chọn ID trong listbox
            selection = listbox.curselection()
            if not selection:
                return
            
            # Lấy ID được chọn cuối cùng
            selected_id = int(listbox.get(selection[-1]))
            
            # Tìm frame có ảnh rõ nhất của person này (bbox lớn nhất)
            best_frame_idx = None
            best_bbox = None
            max_area = 0
            
            for frame_idx, persons_in_frame in self.frame_data.items():
                for person_data in persons_in_frame:
                    if person_data["track_id"] == selected_id:
                        bbox = person_data["bbox"]
                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        if area > max_area:
                            max_area = area
                            best_frame_idx = frame_idx
                            best_bbox = bbox
            
            if best_frame_idx is not None and best_bbox is not None:
                # Đọc frame và cắt bbox
                frame_path = self.frames_list[int(best_frame_idx)]
                frame = cv2.imread(frame_path)
                if frame is not None:
                    x1, y1, x2, y2 = map(int, best_bbox)
                    
                    # Đảm bảo tọa độ nằm trong giới hạn ảnh
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x1 < x2 and y1 < y2:
                        person_img = frame[y1:y2, x1:x2]
                        rgb_person = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
                        
                        # Resize để vừa với canvas
                        canvas_width = preview_canvas.winfo_width()
                        canvas_height = preview_canvas.winfo_height()
                        
                        if canvas_width > 1 and canvas_height > 1:
                            img_h, img_w = rgb_person.shape[:2]
                            ratio = min(canvas_width/img_w, canvas_height/img_h)
                            new_w = int(img_w * ratio)
                            new_h = int(img_h * ratio)
                            
                            resized_person = cv2.resize(rgb_person, (new_w, new_h))
                            photo = ImageTk.PhotoImage(image=Image.fromarray(resized_person))
                            
                            # Cập nhật canvas
                            preview_canvas.delete("all")
                            preview_canvas.create_image(
                                (canvas_width - new_w) // 2,
                                (canvas_height - new_h) // 2,
                                image=photo, anchor=tk.NW
                            )
                            preview_canvas.image = photo  # Giữ reference
                            
                            # Thêm text hiển thị frame_idx
                            preview_canvas.create_text(
                                10, 20,
                                text=f"Frame: {best_frame_idx}",
                                fill="white",
                                anchor=tk.NW
                            )
        
        def do_merge():
            selected = listbox.curselection()
            if len(selected) < 2:
                messagebox.showwarning("Cảnh báo", "Vui lòng chọn ít nhất 2 ID để gộp")
                return
            
            ids_to_merge = [int(listbox.get(i)) for i in selected]
            target_id = min(ids_to_merge)
            
            # Update frame data
            for frame_idx in self.frame_data:
                for person in self.frame_data[frame_idx]:
                    if person["track_id"] in ids_to_merge:
                        person["track_id"] = target_id
            
            # Update person attributes
            for old_id in ids_to_merge:
                if old_id != target_id:
                    if old_id in self.person_attributes:
                        del self.person_attributes[old_id]
            
            # Update UI
            self.update_person_list()
            self.show_frame(self.current_frame_idx)
            dialog.destroy()
            
            messagebox.showinfo("Thành công", f"Đã gộp {len(ids_to_merge)} ID thành ID {target_id}")
        
        # Bind sự kiện chọn trong listbox
        listbox.bind('<<ListboxSelect>>', on_select)
        
        # Thêm nút Merge
        ttk.Button(left_frame, text="Gộp ID đã chọn", command=do_merge).pack(pady=10)

    def delete_id(self):
        # Tạo dialog để chọn ID cần xóa
        dialog = tk.Toplevel(self.root)
        dialog.title("Delete Person ID")
        dialog.geometry("800x500")
        
        # Tạo frame chính chia làm 2 phần
        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Frame bên trái cho danh sách ID
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Frame bên phải cho preview ảnh
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tạo label và listbox trong frame trái
        label = ttk.Label(left_frame, text="Chọn ID để xóa:")
        label.pack(pady=5)
        
        listbox = tk.Listbox(left_frame, selectmode=tk.SINGLE)  # Chỉ cho phép chọn 1 ID
        listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tạo canvas để hiển thị ảnh preview trong frame phải
        ttk.Label(right_frame, text="Preview:").pack(pady=5)
        preview_canvas = tk.Canvas(right_frame, bg="black", width=300, height=400)
        preview_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Thêm các ID vào listbox
        for person_id in sorted(self.person_attributes.keys()):
            listbox.insert(tk.END, str(person_id))
        
        def on_select(event):
            # Xử lý khi chọn ID trong listbox
            selection = listbox.curselection()
            if not selection:
                return
            
            # Lấy ID được chọn
            selected_id = int(listbox.get(selection[0]))
            
            # Tìm frame có ảnh rõ nhất của person này (bbox lớn nhất)
            best_frame_idx = None
            best_bbox = None
            max_area = 0
            
            for frame_idx, persons_in_frame in self.frame_data.items():
                for person_data in persons_in_frame:
                    if person_data["track_id"] == selected_id:
                        bbox = person_data["bbox"]
                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        if area > max_area:
                            max_area = area
                            best_frame_idx = frame_idx
                            best_bbox = bbox
        
            if best_frame_idx is not None and best_bbox is not None:
                # Đọc frame và cắt bbox
                frame_path = self.frames_list[int(best_frame_idx)]
                frame = cv2.imread(frame_path)
                if frame is not None:
                    x1, y1, x2, y2 = map(int, best_bbox)
                    
                    # Đảm bảo tọa độ nằm trong giới hạn ảnh
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x1 < x2 and y1 < y2:
                        person_img = frame[y1:y2, x1:x2]
                        rgb_person = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
                        
                        # Resize để vừa với canvas
                        canvas_width = preview_canvas.winfo_width()
                        canvas_height = preview_canvas.winfo_height()
                        
                        if canvas_width > 1 and canvas_height > 1:
                            img_h, img_w = rgb_person.shape[:2]
                            ratio = min(canvas_width/img_w, canvas_height/img_h)
                            new_w = int(img_w * ratio)
                            new_h = int(img_h * ratio)
                            
                            resized_person = cv2.resize(rgb_person, (new_w, new_h))
                            photo = ImageTk.PhotoImage(image=Image.fromarray(resized_person))
                            
                            # Cập nhật canvas
                            preview_canvas.delete("all")
                            preview_canvas.create_image(
                                (canvas_width - new_w) // 2,
                                (canvas_height - new_h) // 2,
                                image=photo, anchor=tk.NW
                            )
                            preview_canvas.image = photo  # Giữ reference
                            
                            # Thêm text hiển thị frame_idx và số lượng frame có person này
                            frame_count = sum(
                                1 for frame_data in self.frame_data.values()
                                for person in frame_data
                                if person["track_id"] == selected_id
                            )
                            preview_canvas.create_text(
                                10, 20,
                                text=f"Frame: {best_frame_idx}\nXuất hiện trong {frame_count} frames",
                                fill="white",
                                anchor=tk.NW
                            )
        
        def do_delete():
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning("Cảnh báo", "Vui lòng chọn một ID để xóa")
                return
            
            selected_id = int(listbox.get(selection[0]))
            
            # Đếm số frame có person này
            frame_count = sum(
                1 for frame_data in self.frame_data.values()
                for person in frame_data
                if person["track_id"] == selected_id
            )
            
            # Hỏi xác nhận trước khi xóa
            if not messagebox.askyesno("Xác nhận xóa", 
                                     f"Bạn có chắc chắn muốn xóa Person ID {selected_id}?\n"
                                     f"Person này xuất hiện trong {frame_count} frames.\n"
                                     "Hành động này không thể hoàn tác."):
                return
            
            # Xóa person khỏi tất cả các frame
            for frame_idx in self.frame_data:
                self.frame_data[frame_idx] = [
                    person for person in self.frame_data[frame_idx]
                    if person["track_id"] != selected_id
                ]
            
            # Xóa khỏi person_attributes
            if selected_id in self.person_attributes:
                del self.person_attributes[selected_id]
            
            # Update UI
            self.update_person_list()
            self.show_frame(self.current_frame_idx)
            dialog.destroy()
            
            messagebox.showinfo("Thành công", 
                              f"Đã xóa Person ID {selected_id} khỏi {frame_count} frames")
        
        # Bind sự kiện chọn trong listbox
        listbox.bind('<<ListboxSelect>>', on_select)
        
        # Frame chứa các nút
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(pady=10)
        
        # Thêm nút Delete và Cancel
        ttk.Button(button_frame, text="Xóa ID", command=do_delete).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Hủy", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def split_id(self):
        # Tạo dialog để chọn ID cần tách
        dialog = tk.Toplevel(self.root)
        dialog.title("Split Person ID")
        dialog.geometry("1000x600")
        
        # Tạo frame chính chia làm 3 phần
        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Frame bên trái cho danh sách ID
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Frame giữa cho danh sách frames và bounding boxes
        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Frame bên phải cho preview ảnh
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tạo các thành phần trong left_frame
        ttk.Label(left_frame, text="Chọn ID để tách:").pack(pady=5)
        
        # Thêm scrollbar cho id_listbox
        id_scroll = ttk.Scrollbar(left_frame)
        id_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        id_listbox = tk.Listbox(left_frame, height=10, yscrollcommand=id_scroll.set)
        id_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        id_scroll.config(command=id_listbox.yview)
        
        # Tạo các thành phần trong middle_frame
        ttk.Label(middle_frame, text="Danh sách frames và bounding boxes:").pack(pady=5)
        
        # Thêm scrollbar cho frame_listbox
        frame_scroll = ttk.Scrollbar(middle_frame)
        frame_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Listbox hiển thị frame và số lượng bbox trong frame đó
        frame_listbox = tk.Listbox(middle_frame, selectmode=tk.SINGLE, height=10, 
                                  yscrollcommand=frame_scroll.set)
        frame_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        frame_scroll.config(command=frame_listbox.yview)
        
        # Thêm listbox mới để hiển thị các bbox trong frame được chọn
        ttk.Label(middle_frame, text="Bounding boxes trong frame:").pack(pady=5)
        bbox_listbox = tk.Listbox(middle_frame, selectmode=tk.MULTIPLE, height=10)
        bbox_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Thêm frame mới cho các nút điều khiển bbox
        bbox_control_frame = ttk.Frame(middle_frame)
        bbox_control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Tạo canvas để hiển thị ảnh preview trong frame phải
        ttk.Label(right_frame, text="Preview:").pack(pady=5)
        preview_canvas = tk.Canvas(right_frame, bg="black", width=300, height=400)
        preview_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Thêm các nút điều khiển
        ttk.Button(bbox_control_frame, text="Chọn tất cả", 
                  command=lambda: bbox_listbox.select_set(0, tk.END)).pack(side=tk.LEFT, padx=2)
        ttk.Button(bbox_control_frame, text="Bỏ chọn tất cả", 
                  command=lambda: bbox_listbox.selection_clear(0, tk.END)).pack(side=tk.LEFT, padx=2)
        ttk.Button(bbox_control_frame, text="Đảo chọn", 
                  command=lambda: [bbox_listbox.selection_set(i) if i not in bbox_listbox.curselection() 
                                 else bbox_listbox.selection_clear(i) 
                                 for i in range(bbox_listbox.size())]).pack(side=tk.LEFT, padx=2)
        
        # Thêm hướng dẫn sử dụng
        help_text = ttk.Label(middle_frame, text="Hướng dẫn:\n" + 
                             "- Ctrl + Click: Chọn nhiều bbox riêng lẻ\n" + 
                             "- Shift + Click: Chọn một dải bbox\n" + 
                             "- Click đơn: Chọn một bbox", 
                             justify=tk.LEFT)
        help_text.pack(pady=5)
        
        # Biến để lưu trữ thông tin hiện tại
        current_data = {
            'selected_id': None,
            'selected_frame': None,
            'bbox_info': {}  # Lưu thông tin bbox cho mỗi frame
        }
        
        # Điền dữ liệu vào id_listbox
        for person_id in sorted(self.person_attributes.keys()):
            id_listbox.insert(tk.END, person_id)
        
        def update_frame_list(selected_id):
            frame_listbox.delete(0, tk.END)
            current_data['bbox_info'] = {}
            
            # Duyệt qua frame_data để tìm các frame có ID được chọn
            for frame_idx in sorted(self.frame_data.keys()):
                # Đếm số lượng bbox trong frame này có ID được chọn
                bboxes = [person for person in self.frame_data[frame_idx] 
                         if person["track_id"] == selected_id]
                
                if bboxes:
                    frame_text = f"Frame {frame_idx} ({len(bboxes)} bboxes)"
                    frame_listbox.insert(tk.END, frame_text)
                    current_data['bbox_info'][frame_idx] = bboxes
            
            current_data['selected_id'] = selected_id
            bbox_listbox.delete(0, tk.END)  # Xóa danh sách bbox cũ
        
        def update_bbox_list(frame_idx):
            bbox_listbox.delete(0, tk.END)
            if frame_idx in current_data['bbox_info']:
                for i, bbox_data in enumerate(current_data['bbox_info'][frame_idx]):
                    bbox = bbox_data['bbox']
                    bbox_text = f"Bbox {i+1}: [{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]"
                    bbox_listbox.insert(tk.END, bbox_text)
        
        def show_frame_preview(frame_idx, highlight_bbox_indices=None):
            if frame_idx not in self.frame_data:
                return
            
            # Đọc frame
            frame_path = self.frames_list[frame_idx]
            frame = cv2.imread(frame_path)
            if frame is None:
                return
            
            # Vẽ tất cả bbox trong frame với ID được chọn
            if frame_idx in current_data['bbox_info']:
                for i, person_data in enumerate(current_data['bbox_info'][frame_idx]):
                    bbox = person_data['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Màu khác nhau cho bbox được chọn và không được chọn
                    color = (0, 255, 0) if highlight_bbox_indices and i in highlight_bbox_indices else (255, 0, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"Bbox {i+1}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Hiển thị frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize để vừa với canvas
            canvas_width = preview_canvas.winfo_width()
            canvas_height = preview_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                img_h, img_w = rgb_frame.shape[:2]
                ratio = min(canvas_width/img_w, canvas_height/img_h)
                new_w = int(img_w * ratio)
                new_h = int(img_h * ratio)
                
                resized_img = cv2.resize(rgb_frame, (new_w, new_h))
                photo = ImageTk.PhotoImage(image=Image.fromarray(resized_img))
                
                preview_canvas.delete("all")
                preview_canvas.create_image(
                    (canvas_width - new_w) // 2,
                    (canvas_height - new_h) // 2,
                    image=photo, anchor=tk.NW
                )
                preview_canvas.image = photo  # Giữ reference để tránh bị garbage collect
        
        def on_id_select(event):
            selection = id_listbox.curselection()
            if selection:  # Kiểm tra có item được chọn không
                selected_id = int(id_listbox.get(selection[0]))
                update_frame_list(selected_id)
        
        def on_frame_select(event):
            selection = frame_listbox.curselection()
            if selection:  # Kiểm tra có item được chọn không
                frame_text = frame_listbox.get(selection[0])
                frame_idx = int(frame_text.split()[1])
                current_data['selected_frame'] = frame_idx
                update_bbox_list(frame_idx)
                show_frame_preview(frame_idx)
        
        def on_bbox_select(event):
            if current_data['selected_frame'] is not None:
                show_frame_preview(current_data['selected_frame'], bbox_listbox.curselection())
        
        def do_split():
            if not current_data['selected_id'] or current_data['selected_frame'] is None:
                messagebox.showwarning("Cảnh báo", "Vui lòng chọn ID và frame")
                return
            
            selected_bbox_indices = bbox_listbox.curselection()
            if not selected_bbox_indices:
                messagebox.showwarning("Cảnh báo", "Vui lòng chọn ít nhất một bounding box để tách")
                return
            
            # Tạo ID mới
            new_id = max(self.person_attributes.keys()) + 1
            
            # Copy thuộc tính từ ID cũ
            self.person_attributes[new_id] = self.person_attributes[current_data['selected_id']].copy()
            
            # Cập nhật ID cho các bbox được chọn
            frame_idx = current_data['selected_frame']
            if frame_idx in self.frame_data:
                bboxes_in_frame = [person for person in self.frame_data[frame_idx] 
                                 if person["track_id"] == current_data['selected_id']]
                
                for bbox_idx in selected_bbox_indices:
                    if bbox_idx < len(bboxes_in_frame):
                        # Tìm và cập nhật bbox trong frame_data
                        bbox_to_update = bboxes_in_frame[bbox_idx]
                        for person in self.frame_data[frame_idx]:
                            if (person["track_id"] == current_data['selected_id'] and 
                                person["bbox"] == bbox_to_update["bbox"]):
                                person["track_id"] = new_id
                                break
            
            # Update UI
            self.update_person_list()
            self.show_frame(self.current_frame_idx)
            dialog.destroy()
            
            messagebox.showinfo("Thành công", 
                              f"Đã tách {len(selected_bbox_indices)} bounding boxes từ ID "
                              f"{current_data['selected_id']} thành ID mới {new_id}")
        
        # Bind các sự kiện
        id_listbox.bind('<<ListboxSelect>>', on_id_select)
        frame_listbox.bind('<<ListboxSelect>>', on_frame_select)
        bbox_listbox.bind('<<ListboxSelect>>', on_bbox_select)
        
        # Thêm nút tách ID
        ttk.Button(dialog, text="Tách ID đã chọn", command=do_split).pack(pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = PersonAttributeLabelingApp(root)
    root.mainloop()