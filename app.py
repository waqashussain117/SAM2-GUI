import sys
import os
import numpy as np
import torch
from PIL import Image, ImageQt
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QFileDialog,
    QLabel,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QMessageBox,
    QDialog,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
    QDialogButtonBox,
    QMenu,
    QAction,
    QProgressBar,
    QComboBox,
    QSpinBox,
    QCheckBox,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsRectItem,
    QGraphicsItem,
)
try:
    # Try importing PyQt5
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRectF, QPointF
    from PyQt5.QtGui import QCursor, QPixmap, QPen, QColor, QBrush, QPainter
    PYQT_VERSION = 5
except ImportError:
    try:
        # If PyQt5 is not available, try PyQt6
        from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRectF, QPointF
        from PyQt6.QtGui import QCursor, QPixmap, QPen, QColor, QBrush, QPainter
        PYQT_VERSION = 6
    except ImportError:
        raise ImportError("Neither PyQt5 nor PyQt6 is installed. Please install one of them.")
from PyQt5.QtGui import QCursor, QPixmap, QPen, QColor, QBrush, QPainter, QImage
# Import the necessary SAM modules
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)


def pil_to_qimage(pil_image):
    """Convert PIL Image to QImage."""
    if pil_image.mode == "RGBA":
        data = pil_image.tobytes("raw", "RGBA")
        qimage = QImage(data, pil_image.width, pil_image.height, QImage.Format_ARGB32)
    elif pil_image.mode == "RGB":
        data = pil_image.tobytes("raw", "RGB")
        qimage = QImage(data, pil_image.width, pil_image.height, QImage.Format_RGB888)
    elif pil_image.mode == "L":
        data = pil_image.tobytes("raw", "L")
        qimage = QImage(data, pil_image.width, pil_image.height, QImage.Format_Grayscale8)
    else:
        # Convert to RGBA if image mode is not supported
        pil_image = pil_image.convert("RGBA")
        data = pil_image.tobytes("raw", "RGBA")
        qimage = QImage(data, pil_image.width, pil_image.height, QImage.Format_ARGB32)
    return qimage


class MaskRemovalDialog(QDialog):
    """Dialog to remove selected masks."""

    def __init__(self, masks, scores, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Remove Masks")
        self.setMinimumWidth(400)
        self.selected_masks = []

        layout = QVBoxLayout()

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.MultiSelection)

        for idx, score in enumerate(scores):
            item = QListWidgetItem(f"Mask {idx + 1} - Score: {score:.3f}")
            item.setCheckState(Qt.Unchecked)
            self.list_widget.addItem(item)

        layout.addWidget(self.list_widget)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(buttons)
        self.setLayout(layout)

    def get_selected_masks(self):
        selected = []
        for index in range(self.list_widget.count()):
            item = self.list_widget.item(index)
            if item.checkState() == Qt.Checked:
                selected.append(index)
        return selected


class PointRemovalDialog(QDialog):
    """Dialog to remove selected points."""

    def __init__(self, points, labels, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Remove Points")
        self.setMinimumWidth(400)
        self.selected_points = []

        layout = QVBoxLayout()

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.MultiSelection)

        for idx, (point, label) in enumerate(zip(points, labels)):
            label_text = "Foreground" if label == 1 else "Background"
            item = QListWidgetItem(f"Point {idx + 1}: ({point[0]}, {point[1]}) - {label_text}")
            item.setCheckState(Qt.Unchecked)
            self.list_widget.addItem(item)

        layout.addWidget(self.list_widget)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(buttons)
        self.setLayout(layout)

    def get_selected_points(self):
        selected = []
        for index in range(self.list_widget.count()):
            item = self.list_widget.item(index)
            if item.checkState() == Qt.Checked:
                selected.append(index)
        return selected


class LineRemovalDialog(QDialog):
    """Dialog to remove selected lines."""

    def __init__(self, lines, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Remove Lines")
        self.setMinimumWidth(400)
        self.selected_lines = []

        layout = QVBoxLayout()

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.MultiSelection)

        for idx, line in enumerate(lines):
            item = QListWidgetItem(
                f"Line {idx + 1}: Start({line[0][0]}, {line[0][1]}), End({line[1][0]}, {line[1][1]})"
            )
            item.setCheckState(Qt.Unchecked)
            self.list_widget.addItem(item)

        layout.addWidget(self.list_widget)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(buttons)
        self.setLayout(layout)

    def get_selected_lines(self):
        selected = []
        for index in range(self.list_widget.count()):
            item = self.list_widget.item(index)
            if item.checkState() == Qt.Checked:
                selected.append(index)
        return selected


class ImageCanvas(QGraphicsView):
    """Canvas to display images and annotations using QGraphicsView."""

    point_selected = pyqtSignal(int)  # Emit index of selected point

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = None

        # Annotation items
        self.point_items = []
        self.line_items = []
        self.box_items = []
        self.mask_items = []

        # Annotations data
        self.parent = parent
        self.selected_point_index = None
        self.dragging_point = False
        self.point_pick_threshold = 10  # Pixel distance for selecting a point

        # Line drawing state
        self.drawing_line = False
        self.temp_line = None

        # Zoom parameters
        self.scale_factor = 1.0

        # Set render hints for better quality
        if PYQT_VERSION == 5:
            self.setRenderHints(self.renderHints() | QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        elif PYQT_VERSION == 6:
            self.setRenderHints(self.renderHints() | QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)

        # Enable mouse tracking
        self.setMouseTracking(True)

    def display_image(self, image):
        """Displays the image on the canvas."""
        self.scene.clear()
        self.point_items.clear()
        self.line_items.clear()
        self.box_items.clear()
        self.mask_items.clear()

        self.image = image

        # Convert PIL Image to QImage using the helper function
        qimage = pil_to_qimage(image)
        if qimage.isNull():
            QMessageBox.critical(self, "Error", "Failed to convert PIL Image to QImage.")
            return

        # Convert QImage to QPixmap
        pixmap = QPixmap.fromImage(qimage)
        if pixmap.isNull():
            QMessageBox.critical(self, "Error", "Failed to convert QImage to QPixmap.")
            return

        self.pixmap_item = self.scene.addPixmap(pixmap)

        # **Fixed Line: Replace self.pixmap with self.pixmap_item.pixmap()**
        self.setSceneRect(QRectF(self.pixmap_item.pixmap().rect()))
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        self.scale_factor = 1.0

    def update_annotations(self, points, labels, lines, boxes, masks):
        """Updates all annotations on the canvas."""
        # Clear existing annotations
        for item in self.point_items + self.line_items + self.box_items + self.mask_items:
            self.scene.removeItem(item)
        self.point_items.clear()
        self.line_items.clear()
        self.box_items.clear()
        self.mask_items.clear()

        # Display masks first (so they appear below other annotations)
        for mask in masks:
            mask_pixmap = self.create_mask_pixmap(mask)
            mask_item = self.scene.addPixmap(mask_pixmap)
            mask_item.setZValue(1)  # Set below points, lines, boxes
            self.mask_items.append(mask_item)

        # Display boxes
        for box in boxes:
            rect_item = self.create_box_item(box)
            self.scene.addItem(rect_item)
            self.box_items.append(rect_item)

        # Display lines
        for line in lines:
            line_item = self.create_line_item(line)
            self.scene.addItem(line_item)
            self.line_items.append(line_item)

        # Display points
        for idx, (point, label) in enumerate(zip(points, labels)):
            point_item = self.create_point_item(point, label, idx)
            self.scene.addItem(point_item)
            self.point_items.append(point_item)

    def create_point_item(self, point, label, idx):
        """Creates a graphical point item."""
        x, y = point
        radius = 5
        ellipse = QGraphicsEllipseItem(x - radius, y - radius, 2 * radius, 2 * radius)
        color = QColor(0, 255, 0) if label == 1 else QColor(255, 0, 0)
        ellipse.setBrush(QBrush(color))
        ellipse.setPen(QPen(Qt.white))
        ellipse.setZValue(2)
        ellipse.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable | QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        ellipse.setData(0, idx)  # Store index
        return ellipse

    def create_line_item(self, line):
        """Creates a graphical line item."""
        (x0, y0), (x1, y1) = line
        line_item = QGraphicsLineItem(x0, y0, x1, y1)
        pen = QPen(Qt.yellow, 2)
        line_item.setPen(pen)
        line_item.setZValue(2)
        return line_item

    def create_box_item(self, box):
        """Creates a graphical box item."""
        x0, y0, x1, y1 = box
        rect_item = QGraphicsRectItem(x0, y0, x1 - x0, y1 - y0)
        pen = QPen(Qt.green, 2)
        rect_item.setPen(pen)
        rect_item.setZValue(2)
        return rect_item

    def create_mask_pixmap(self, mask):
        """Creates a semi-transparent pixmap from a binary mask."""
        try:
            mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
            colored_mask = Image.new("RGBA", mask_image.size, (30, 144, 255, 100))  # Increased alpha for visibility
            mask_overlay = Image.composite(
                colored_mask,
                Image.new("RGBA", mask_image.size, (0, 0, 0, 0)),
                mask_image
            )

            # Use the pil_to_qimage function for conversion
            qimage = pil_to_qimage(mask_overlay)
            if qimage.isNull():
                QMessageBox.critical(self, "Error", "Failed to convert mask overlay to QImage.")
                return QPixmap()

            return QPixmap.fromImage(qimage)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create mask pixmap:\n{str(e)}")
            print(f"Error in create_mask_pixmap: {e}")
            return QPixmap()

    def mousePressEvent(self, event):
        """Handles mouse press events."""
        if event.button() == Qt.LeftButton:
            pos = self.mapToScene(event.position().toPoint()) if PYQT_VERSION == 6 else self.mapToScene(event.pos())
            x, y = int(pos.x()), int(pos.y())

            modifiers = QApplication.keyboardModifiers()
            if modifiers == Qt.ControlModifier:
                label = 1  # Foreground
            elif modifiers == Qt.ShiftModifier:
                label = 0  # Background
            else:
                label = None

            if label is not None:
                # Add point
                self.parent.point_coords.append([x, y])
                self.parent.point_labels.append(label)
                self.parent.add_action_to_undo("add_point", ([x, y], label))
                self.parent.update_canvas()
                self.parent.remove_mask_btn.setEnabled(True)
                self.parent.remove_point_btn.setEnabled(True)
                # Trigger real-time mask update
                self.parent.generate_mask(real_time=True)
            elif self.parent.draw_line_mode:
                if not self.parent.start_point:
                    self.parent.start_point = (x, y)
                else:
                    end_point = (x, y)
                    line = [self.parent.start_point, end_point]
                    self.parent.lines.append(line)
                    self.parent.add_action_to_undo("add_line", line)
                    self.parent.start_point = None
                    self.parent.update_canvas()
                    self.parent.remove_mask_btn.setEnabled(True)
                    self.parent.remove_line_btn.setEnabled(True)
                    # Trigger real-time mask update
                    self.parent.generate_mask(real_time=True)
            else:
                # Check if a point is clicked for dragging
                index = self.get_point_index_near(x, y)
                if index is not None:
                    self.selected_point_index = index
                    self.dragging_point = True
        elif event.button() == Qt.RightButton:
            # Open context menu if right-clicking on a point
            pos = self.mapToScene(event.position().toPoint()) if PYQT_VERSION == 6 else self.mapToScene(event.pos())
            x, y = int(pos.x()), int(pos.y())
            idx = self.get_point_index_near(x, y)
            if idx is not None:
                self.open_context_menu(event, idx)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handles mouse move events."""
        pos = self.mapToScene(event.position().toPoint()) if PYQT_VERSION == 6 else self.mapToScene(event.pos())
        x, y = int(pos.x()), int(pos.y())

        idx = self.get_point_index_near(x, y)
        if idx is not None and not self.dragging_point:
            self.setCursor(Qt.OpenHandCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

        if self.dragging_point and self.selected_point_index is not None:
            # Update point position
            self.parent.point_coords[self.selected_point_index] = [x, y]
            self.parent.update_canvas()
            # Trigger real-time mask update
            self.parent.generate_mask(real_time=True)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handles mouse release events."""
        if event.button() == Qt.LeftButton and self.dragging_point:
            self.dragging_point = False
            self.selected_point_index = None
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        """Handles zooming with the mouse wheel."""
        if PYQT_VERSION == 6:
            angle = event.angleDelta().y()
        else:
            angle = event.angleDelta().y()

        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        if angle > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor

        self.scale(zoom_factor, zoom_factor)
        self.scale_factor *= zoom_factor

    def get_point_index_near(self, x, y):
        """Returns the index of a point near the given coordinates."""
        for i, (px, py) in enumerate(self.parent.point_coords):
            distance = np.hypot(px - x, py - y)
            if distance < self.point_pick_threshold:
                return i
        return None

    def open_context_menu(self, event, point_index):
        """Opens a context menu for the point at the given index."""
        menu = QMenu(self)
        delete_action = QAction("Delete Point", self)
        delete_action.triggered.connect(lambda: self.delete_point(point_index))
        change_label_action = QAction("Toggle Point Label", self)
        change_label_action.triggered.connect(lambda: self.toggle_point_label(point_index))
        menu.addAction(delete_action)
        menu.addAction(change_label_action)
        menu.exec_(QCursor.pos())

    def delete_point(self, point_index):
        point = self.parent.point_coords.pop(point_index)
        label = self.parent.point_labels.pop(point_index)
        self.parent.add_action_to_undo("delete_point", (point_index, point, label))
        self.parent.update_canvas()
        if not self.parent.point_coords:
            self.parent.remove_point_btn.setEnabled(False)
        # Trigger real-time mask update
        self.parent.generate_mask(real_time=True)

    def toggle_point_label(self, point_index):
        current_label = self.parent.point_labels[point_index]
        self.parent.point_labels[point_index] = 1 - current_label  # Toggle between 0 and 1
        self.parent.update_canvas()
        # Trigger real-time mask update
        self.parent.generate_mask(real_time=True)


class MaskGenerationThread(QThread):
    """Thread to handle mask generation without freezing the UI."""

    mask_generated = pyqtSignal(object, object)  # masks, scores
    error_occurred = pyqtSignal(str)
    progress_update = pyqtSignal(int)

    def __init__(
        self,
        predictor,
        image_features,
        orig_hw,
        point_coords=None,
        point_labels=None,
        lines=None,
        boxes=None,
        multimask_output=True,
    ):
        super().__init__()
        self.predictor = predictor
        self.image_features = image_features  # Store image features
        self.orig_hw = orig_hw  # Store original image size
        self.point_coords = point_coords
        self.point_labels = point_labels
        self.lines = lines
        self.boxes = boxes
        self.multimask_output = multimask_output
        self.is_running = True

    def run(self):
        try:
            # Set image embeddings to avoid recomputing them
            self.predictor.set_image_embedding(self.image_features, self.orig_hw)

            # Convert lines to points or boxes as per SAM2 requirements
            if self.lines:
                # Sample points along each line
                sampled_points = []
                sampled_labels = []
                for line in self.lines:
                    (x0, y0), (x1, y1) = line
                    num_samples = max(abs(x1 - x0), abs(y1 - y0)) // 10  # Sample every ~10 pixels
                    for i in range(num_samples + 1):
                        x = int(x0 + (x1 - x0) * i / num_samples)
                        y = int(y0 + (y1 - y0) * i / num_samples)
                        sampled_points.append([y, x])
                        sampled_labels.append(1)  # Foreground label
                if self.point_coords is not None:
                    self.point_coords = (
                        np.vstack([self.point_coords, np.array(sampled_points)])
                        if len(sampled_points) > 0
                        else self.point_coords
                    )
                else:
                    self.point_coords = np.array(sampled_points) if len(sampled_points) > 0 else None
                if self.point_labels is not None:
                    self.point_labels = (
                        np.hstack([self.point_labels, np.array(sampled_labels)])
                        if len(sampled_labels) > 0
                        else self.point_labels
                    )
                else:
                    self.point_labels = (
                        np.array(sampled_labels) if len(sampled_labels) > 0 else None
                    )

            # Move data to device
            device = self.predictor.device
            if self.point_coords is not None:
                point_coords = torch.tensor(self.point_coords, device=device)
            else:
                point_coords = None
            if self.point_labels is not None:
                point_labels = torch.tensor(self.point_labels, device=device)
            else:
                point_labels = None
            if self.boxes is not None:
                boxes = torch.tensor(self.boxes, device=device)
            else:
                boxes = None

            # Generate mask
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=boxes,
                multimask_output=self.multimask_output,
                return_logits=True,
            )

            # Post-processing: Apply morphological operations
            processed_masks = []
            for mask in masks:
                binary_mask = mask > self.predictor.mask_threshold
                processed_mask = self.postprocess_mask(binary_mask)
                processed_masks.append(processed_mask)

            masks = np.array(processed_masks)

            self.mask_generated.emit(masks, scores)
        except Exception as e:
            error_msg = f"An error occurred during mask generation: {str(e)}"
            self.error_occurred.emit(error_msg)

    def postprocess_mask(self, mask):
        """Applies post-processing to the mask."""
        from skimage.morphology import remove_small_objects, remove_small_holes

        mask = remove_small_objects(mask, min_size=64)
        mask = remove_small_holes(mask, area_threshold=64)
        return mask

    def stop(self):
        self.is_running = False
        self.terminate()


class SAM2GUI(QMainWindow):
    """Main GUI application for SAM2-based image segmentation."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM2 Image Segmentation Tool")
        self.setGeometry(100, 100, 1600, 900)

        # Initialize SAM2 model
        self.init_sam2()

        # Initialize UI
        self.init_ui()

        # Variables to store image and annotations
        self.image = None
        self.point_coords = []
        self.point_labels = []
        self.lines = []
        self.boxes = []
        self.masks = []
        self.scores = []
        self.image_embeddings = None

        # Undo/Redo stacks
        self.undo_stack = []
        self.redo_stack = []

    def init_sam2(self):
        """Initializes the SAM2 model."""
        # Set device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        # Load SAM2 model
        # Provide options for different model sizes
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_types = {
            "Tiny": (
                os.path.join(base_dir, "checkpoints/sam2.1_hiera_tiny.pt"),
                os.path.join(base_dir, "sam2/configs/sam2.1/sam2.1_hiera_t.yaml"),
            ),
            "Small": (
                os.path.join(base_dir, "checkpoints/sam2.1_hiera_small.pt"),
                os.path.join(base_dir, "sam2/configs/sam2.1/sam2.1_hiera_s.yaml"),
            ),
            "Base+": (
                os.path.join(base_dir, "checkpoints/sam2.1_hiera_base_plus.pt"),
                os.path.join(base_dir, "sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"),
            ),
            "Large": (
                os.path.join(base_dir, "checkpoints/sam2.1_hiera_large.pt"),
                os.path.join(base_dir, "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"),
            ),
        }

        # Default to Base+ model
        model_type = "Base+"
        sam2_checkpoint, model_cfg = self.model_types[model_type]

        # Check if files exist
        if not os.path.exists(sam2_checkpoint) or not os.path.exists(model_cfg):
            QMessageBox.critical(
                self,
                "Error",
                f"Model files not found. Please ensure that the checkpoint and config files exist at the specified paths.",
            )
            sys.exit(1)

        self.sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)

    def init_ui(self):
        """Sets up the user interface."""
        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layouts
        main_layout = QHBoxLayout()
        control_layout = QVBoxLayout()
        image_layout = QVBoxLayout()

        # Buttons
        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.clicked.connect(self.upload_image)

        self.draw_line_btn = QPushButton("Draw Line")
        self.draw_line_btn.setCheckable(True)
        self.draw_line_btn.clicked.connect(self.toggle_draw_line_mode)

        self.remove_point_btn = QPushButton("Remove Point")
        self.remove_point_btn.clicked.connect(self.remove_point)
        self.remove_point_btn.setEnabled(False)

        self.remove_line_btn = QPushButton("Remove Line")
        self.remove_line_btn.clicked.connect(self.remove_line)
        self.remove_line_btn.setEnabled(False)

        self.generate_mask_btn = QPushButton("Generate Mask")
        self.generate_mask_btn.clicked.connect(self.generate_mask)
        self.generate_mask_btn.setEnabled(False)

        self.remove_mask_btn = QPushButton("Remove Mask")
        self.remove_mask_btn.clicked.connect(self.remove_mask)
        self.remove_mask_btn.setEnabled(False)

        self.clear_btn = QPushButton("Clear Annotations")
        self.clear_btn.clicked.connect(self.clear_annotations)
        self.clear_btn.setEnabled(False)

        # Undo/Redo Buttons
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.clicked.connect(self.undo)
        self.undo_btn.setEnabled(False)
        self.redo_btn = QPushButton("Redo")
        self.redo_btn.clicked.connect(self.redo)
        self.redo_btn.setEnabled(False)

        # Model Type Selection
        self.model_type_label = QLabel("Model Type:")
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(self.model_types.keys())
        self.model_type_combo.setCurrentText("Base+")
        self.model_type_combo.currentIndexChanged.connect(self.change_model_type)

        # Batch Size Selection
        self.batch_size_label = QLabel("Batch Size:")
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setMinimum(1)
        self.batch_size_spin.setMaximum(64)
        self.batch_size_spin.setValue(1)

        # Multimask Output Checkbox
        self.multimask_checkbox = QCheckBox("Enable Multimask Output")
        self.multimask_checkbox.setChecked(True)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

        # Instructions Label
        self.instructions_label = QLabel(
            "Instructions:\n"
            "- Hold 'Ctrl' and click to add Foreground Point\n"
            "- Hold 'Shift' and click to add Background Point\n"
            "- Use mouse wheel to zoom\n"
            "- Right-click on point to delete or toggle label\n"
            "- Click 'Draw Line' to toggle line drawing mode"
        )

        # Adding widgets to control layout
        control_layout.addWidget(self.upload_btn)
        control_layout.addWidget(self.draw_line_btn)
        control_layout.addWidget(self.remove_point_btn)
        control_layout.addWidget(self.remove_line_btn)
        control_layout.addWidget(self.generate_mask_btn)
        control_layout.addWidget(self.remove_mask_btn)
        control_layout.addWidget(self.undo_btn)
        control_layout.addWidget(self.redo_btn)
        control_layout.addWidget(self.clear_btn)
        control_layout.addWidget(self.progress_bar)
        control_layout.addWidget(self.model_type_label)
        control_layout.addWidget(self.model_type_combo)
        control_layout.addWidget(self.batch_size_label)
        control_layout.addWidget(self.batch_size_spin)
        control_layout.addWidget(self.multimask_checkbox)
        control_layout.addWidget(self.instructions_label)
        control_layout.addStretch()

        # Image Canvas
        self.canvas = ImageCanvas(self)
        image_layout.addWidget(self.canvas)

        # Set layouts
        main_layout.addLayout(control_layout)
        main_layout.addLayout(image_layout)
        central_widget.setLayout(main_layout)

        # Event Flags
        self.draw_line_mode = False

        self.start_point = None  # For line drawing

    def change_model_type(self):
        """Changes the SAM2 model type."""
        model_type = self.model_type_combo.currentText()
        sam2_checkpoint, model_cfg = self.model_types[model_type]

        # Check if files exist
        if not os.path.exists(sam2_checkpoint) or not os.path.exists(model_cfg):
            QMessageBox.critical(
                self,
                "Error",
                f"Model files not found for {model_type}. Please ensure that the checkpoint and config files exist.",
            )
            return

        # Load the new model
        self.sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)
        if self.image is not None:
            self.predictor.set_image(np.array(self.image).astype(np.float32) / 255.0)
            self.image_embeddings, self.orig_hw = self.predictor.get_image_embedding()

    def upload_image(self):
        """Handles image uploading."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)", options=options
        )
        if file_path:
            try:
                image = Image.open(file_path).convert("RGB")  # Ensure RGBA mode
                # Optional: Resize image if too large
                max_size = (1024, 1024)
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
                self.image = image
                self.canvas.display_image(self.image)
                self.predictor.set_image(np.array(self.image).astype(np.float32) / 255.0)
                self.image_embeddings, self.orig_hw = self.predictor.get_image_embedding()
                self.generate_mask_btn.setEnabled(True)
                self.clear_btn.setEnabled(True)
                self.remove_mask_btn.setEnabled(False)
                self.remove_point_btn.setEnabled(False)
                self.remove_line_btn.setEnabled(False)
                # Reset annotations
                self.point_coords = []
                self.point_labels = []
                self.lines = []
                self.boxes = []
                self.masks = []
                self.scores = []
                # Clear undo/redo stacks
                self.undo_stack.clear()
                self.redo_stack.clear()
                self.undo_btn.setEnabled(False)
                self.redo_btn.setEnabled(False)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to process the image:\n{str(e)}")
                print(f"Error processing image: {e}")

    def toggle_draw_line_mode(self):
        """Toggles the line drawing mode."""
        if self.draw_line_btn.isChecked():
            self.draw_line_mode = True
        else:
            self.draw_line_mode = False

    def update_canvas(self):
        """Updates the canvas with current annotations."""
        self.canvas.update_annotations(
            self.point_coords, self.point_labels, self.lines, self.boxes, self.masks
        )

    def generate_mask(self, real_time=False):
        """Initiates mask generation based on current annotations."""
        if self.image is None:
            QMessageBox.warning(self, "No Image", "Please upload an image first.")
            return

        if not (self.point_coords or self.lines or self.boxes):
            QMessageBox.warning(self, "No Prompts", "Please add points, lines, or boxes as prompts.")
            return

        # If real_time is False, disable the generate mask button to prevent multiple clicks
        if not real_time:
            self.generate_mask_btn.setEnabled(False)
            self.remove_mask_btn.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

        # Start mask generation thread
        self.thread = MaskGenerationThread(
            predictor=self.predictor,
            image_features=self.image_embeddings,  # Pass image_embeddings
            orig_hw=self.orig_hw,  # Pass orig_hw
            point_coords=np.array(self.point_coords) if self.point_coords else None,
            point_labels=np.array(self.point_labels) if self.point_labels else None,
            lines=self.lines,
            boxes=np.array(self.boxes) if self.boxes else None,
            multimask_output=self.multimask_checkbox.isChecked(),
        )
        self.thread.mask_generated.connect(self.on_mask_generated)
        self.thread.error_occurred.connect(self.on_mask_error)
        if not real_time:
            self.thread.finished.connect(self.on_mask_generation_finished)
        self.thread.start()

    def on_mask_generated(self, masks, scores):
        """Handles the generated masks."""
        try:
            # Sort masks by scores
            sorted_ind = np.argsort(scores)[::-1].tolist()  # Convert to list
            if not sorted_ind:
                QMessageBox.warning(self, "No Masks", "No masks were generated.")
                return
            top_mask = masks[sorted_ind[0]]
            top_score = scores[sorted_ind[0]]
            self.masks = [top_mask]  # Replace existing masks with the top mask
            self.scores = [top_score]
            self.update_canvas()
            self.remove_mask_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error processing masks: {str(e)}")
            print(f"Error processing masks: {e}")

    def on_mask_error(self, error_msg):
        """Handles errors during mask generation."""
        QMessageBox.critical(self, "Error", error_msg)
        print(error_msg)

    def on_mask_generation_finished(self):
        """Re-enables the generate mask button after mask generation."""
        self.generate_mask_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

    def clear_annotations(self):
        """Clears all annotations from the canvas."""
        self.point_coords = []
        self.point_labels = []
        self.lines = []
        self.boxes = []
        self.masks = []
        self.scores = []
        self.update_canvas()
        self.remove_mask_btn.setEnabled(False)
        self.remove_point_btn.setEnabled(False)
        self.remove_line_btn.setEnabled(False)
        # Clear undo/redo stacks
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.undo_btn.setEnabled(False)
        self.redo_btn.setEnabled(False)

    def remove_mask(self):
        """Removes selected masks."""
        if not self.masks:
            QMessageBox.information(self, "No Masks", "There are no masks to remove.")
            return

        dialog = MaskRemovalDialog(self.masks, self.scores, self)
        if dialog.exec_() == QDialog.Accepted:
            selected_indices = dialog.get_selected_masks()
            if selected_indices:
                # Since we maintain only one mask, clear it
                self.masks = []
                self.scores = []
                self.update_canvas()
                self.remove_mask_btn.setEnabled(False)
            else:
                QMessageBox.information(self, "No Selection", "No masks were selected for removal.")

    def remove_point(self):
        """Removes selected points."""
        if not self.point_coords:
            QMessageBox.information(self, "No Points", "There are no points to remove.")
            return

        dialog = PointRemovalDialog(self.point_coords, self.point_labels, self)
        if dialog.exec_() == QDialog.Accepted:
            selected_indices = dialog.get_selected_points()
            if selected_indices:
                # Remove points based on selected indices
                for index in sorted(selected_indices, reverse=True):
                    point = self.point_coords.pop(index)
                    label = self.point_labels.pop(index)
                    self.add_action_to_undo("delete_point", (index, point, label))
                self.update_canvas()
                if not self.point_coords:
                    self.remove_point_btn.setEnabled(False)
                # Trigger real-time mask update
                self.generate_mask(real_time=True)
            else:
                QMessageBox.information(self, "No Selection", "No points were selected for removal.")

    def remove_line(self):
        """Removes selected lines."""
        if not self.lines:
            QMessageBox.information(self, "No Lines", "There are no lines to remove.")
            return

        dialog = LineRemovalDialog(self.lines, self)
        if dialog.exec_() == QDialog.Accepted:
            selected_indices = dialog.get_selected_lines()
            if selected_indices:
                # Remove lines based on selected indices
                for index in sorted(selected_indices, reverse=True):
                    line = self.lines.pop(index)
                    self.add_action_to_undo("delete_line", (index, line))
                self.update_canvas()
                if not self.lines:
                    self.remove_line_btn.setEnabled(False)
                # Trigger real-time mask update
                self.generate_mask(real_time=True)
            else:
                QMessageBox.information(self, "No Selection", "No lines were selected for removal.")

    def add_action_to_undo(self, action, data):
        self.undo_stack.append((action, data))
        self.undo_btn.setEnabled(True)
        self.redo_stack.clear()
        self.redo_btn.setEnabled(False)

    def undo(self):
        if not self.undo_stack:
            return
        action, data = self.undo_stack.pop()
        self.redo_stack.append((action, data))
        self.redo_btn.setEnabled(True)
        # Implement the undo action based on the action type
        if action == "add_point":
            self.point_coords.pop()
            self.point_labels.pop()
        elif action == "delete_point":
            idx, point, label = data
            self.point_coords.insert(idx, point)
            self.point_labels.insert(idx, label)
        elif action == "add_line":
            self.lines.pop()
        elif action == "delete_line":
            idx, line = data
            self.lines.insert(idx, line)
        # Update the canvas
        self.update_canvas()
        # Trigger real-time mask update
        self.generate_mask(real_time=True)
        if not self.undo_stack:
            self.undo_btn.setEnabled(False)

    def redo(self):
        if not self.redo_stack:
            return
        action, data = self.redo_stack.pop()
        self.undo_stack.append((action, data))
        self.undo_btn.setEnabled(True)
        # Implement the redo action based on the action type
        if action == "add_point":
            point, label = data
            self.point_coords.append(point)
            self.point_labels.append(label)
        elif action == "delete_point":
            idx = data[0]
            self.point_coords.pop(idx)
            self.point_labels.pop(idx)
        elif action == "add_line":
            line = data
            self.lines.append(line)
        elif action == "delete_line":
            idx = data[0]
            self.lines.pop(idx)
        # Update the canvas
        self.update_canvas()
        # Trigger real-time mask update
        self.generate_mask(real_time=True)
        if not self.redo_stack:
            self.redo_btn.setEnabled(False)


def main():
    """Main function to run the application."""
    app = QApplication(sys.argv)
    window = SAM2GUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
