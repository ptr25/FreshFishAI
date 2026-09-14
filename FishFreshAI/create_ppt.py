"""Generate PowerPoint presentation for FishFreshAI project."""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import os

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# ── Color scheme ──────────────────────────────────────────────────────────────
BLUE = RGBColor(0, 102, 204)
DARK_BLUE = RGBColor(0, 51, 102)
WHITE = RGBColor(255, 255, 255)
BLACK = RGBColor(0, 0, 0)
LIGHT_GRAY = RGBColor(240, 240, 240)
GREEN = RGBColor(0, 153, 76)
RED = RGBColor(204, 0, 0)
ORANGE = RGBColor(255, 153, 0)

RESULTS_DIR = '/Users/pranathi_tr/Desktop/ai ml/FishFreshAI/results'

def add_bg(slide, color=DARK_BLUE):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_text_box(slide, left, top, width, height, text, font_size=18, color=BLACK, bold=False, alignment=PP_ALIGN.LEFT):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.alignment = alignment
    return txBox

def add_bullet_slide(slide, items, left=1, top=2, width=11, font_size=20, color=BLACK):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(5))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.space_after = Pt(8)
        p.level = 0

def add_title_bar(slide, title):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.2))
    shape.fill.solid()
    shape.fill.fore_color.rgb = BLUE
    shape.line.fill.background()
    add_text_box(slide, 0.5, 0.15, 12, 1, title, font_size=36, color=WHITE, bold=True)

def add_image_safe(slide, path, left, top, width=None, height=None):
    if os.path.exists(path):
        if width and height:
            slide.shapes.add_picture(path, Inches(left), Inches(top), Inches(width), Inches(height))
        elif width:
            slide.shapes.add_picture(path, Inches(left), Inches(top), width=Inches(width))
        elif height:
            slide.shapes.add_picture(path, Inches(left), Inches(top), height=Inches(height))
        else:
            slide.shapes.add_picture(path, Inches(left), Inches(top))

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 1: Title
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, DARK_BLUE)
add_text_box(slide, 1, 1.5, 11, 1.5, "FISH FRESHNESS DETECTION", font_size=48, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
add_text_box(slide, 1, 3.2, 11, 1, "USING DEEP LEARNING", font_size=36, color=ORANGE, bold=True, alignment=PP_ALIGN.CENTER)
add_text_box(slide, 1, 4.5, 11, 0.6, "EfficientNet-B0 Transfer Learning", font_size=24, color=WHITE, alignment=PP_ALIGN.CENTER)
add_text_box(slide, 1, 5.5, 11, 0.5, "[Your Name] | [Your Roll No] | [Your College]", font_size=18, color=WHITE, alignment=PP_ALIGN.CENTER)
add_text_box(slide, 1, 6.2, 11, 0.5, "[Department] | [Month Year]", font_size=16, color=WHITE, alignment=PP_ALIGN.CENTER)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 2: Abstract
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "Abstract")
add_bullet_slide(slide, [
    "Fish freshness is critical for food quality and safety",
    "Manual inspection is time-consuming and inconsistent",
    "We built an AI system using Deep Learning to classify fish freshness",
    "Uses EfficientNet-B0 with Transfer Learning (pretrained on ImageNet)",
    "Input: Fish eye/gill image → Output: Fresh / Non-Fresh",
    "Achieved 99.62% accuracy on test dataset",
    "Automates the freshness detection process using computer vision"
], font_size=20)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 3: Introduction
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "Introduction")
add_bullet_slide(slide, [
    "Fish freshness is vital for maintaining food quality and safety",
    "Currently checked manually by looking at eyes, skin, gills, and appearance",
    "Manual inspection requires experience and gives inconsistent results",
    "Deep Learning and Computer Vision can automate this process",
    "CNNs (Convolutional Neural Networks) are widely used for image classification",
    "Transfer Learning allows us to use pretrained models for better results",
    "Our system analyzes fish images and predicts freshness automatically"
], font_size=20)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 4: Problem Definition
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "Problem Definition")
add_text_box(slide, 1, 2.5, 11, 2,
    "To develop an image-based deep learning system that can\nclassify fish according to their freshness using images,\nproviding a quick and automated prediction.",
    font_size=28, color=DARK_BLUE, alignment=PP_ALIGN.CENTER)
add_bullet_slide(slide, [
    "Input: Fish eye or gill image",
    "Output: Fresh or Non-Fresh classification with confidence score",
    "Goal: Replace manual inspection with automated AI-based detection"
], top=5, font_size=20)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 5: Objectives
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "Objectives")
add_bullet_slide(slide, [
    "Collect a suitable dataset of fish images",
    "Classify images into Fresh and Non-Fresh categories",
    "Preprocess images (resize, normalize, augment)",
    "Train a deep learning model using Transfer Learning",
    "Handle class imbalance using class weights",
    "Test the model on unseen images",
    "Predict freshness of new fish images",
    "Evaluate performance using accuracy, precision, recall, F1-score"
], font_size=20)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 6: Existing System
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "Existing System")
add_text_box(slide, 1, 1.5, 11, 0.6, "Current Method: Manual Visual Inspection", font_size=24, color=DARK_BLUE, bold=True)
add_text_box(slide, 1, 2.2, 11, 0.6, "Limitations:", font_size=22, color=RED, bold=True)
add_bullet_slide(slide, [
    "Requires human experience and expertise",
    "Results vary between individuals",
    "Time-consuming process",
    "Difficult to maintain consistent evaluation",
    "Cannot handle large volumes of fish",
    "Subjective and prone to human error"
], top=3, font_size=20)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 7: Proposed System
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "Proposed System")
add_text_box(slide, 1, 1.5, 11, 0.8, "AI-Based Fish Freshness Detection", font_size=24, color=DARK_BLUE, bold=True, alignment=PP_ALIGN.CENTER)

# Flow diagram using text boxes
flow_items = ["Fish Image", "Preprocessing", "EfficientNet-B0\nModel", "Classification", "Fresh /\nNon-Fresh"]
colors = [BLUE, GREEN, ORANGE, BLUE, GREEN]
for i, (item, clr) in enumerate(zip(flow_items, colors)):
    x = 0.8 + i * 2.5
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(3), Inches(2), Inches(1.5))
    shape.fill.solid()
    shape.fill.fore_color.rgb = clr
    shape.line.fill.background()
    tf = shape.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = item
    p.font.size = Pt(16)
    p.font.color.rgb = WHITE
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER
    if i < len(flow_items) - 1:
        arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + 2.1), Inches(3.5), Inches(0.3), Inches(0.4))
        arrow.fill.solid()
        arrow.fill.fore_color.rgb = BLACK
        arrow.line.fill.background()

add_bullet_slide(slide, [
    "Uses EfficientNet-B0 pretrained on ImageNet (1.4M images)",
    "Transfer Learning: backbone frozen, custom head trained",
    "Handles class imbalance with automatic weight computation",
    "Auto-detects device: MPS (Apple Silicon) / CUDA / CPU"
], top=5.2, font_size=18)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 8: Dataset
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "Dataset")
add_bullet_slide(slide, [
    "Dataset: JIS Fish Classification Dataset (Kaggle)",
    "Contains images of fish eyes and gills",
    "Two categories: Fresh and Non-Fresh",
    "Total images: 3,926",
    "Training: 2,510 images (80%)",
    "Validation: 629 images (10%)",
    "Testing: 787 images (10%)",
    "Class distribution: 1,060 Fresh / 2,866 Non-Fresh"
], font_size=20)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 9: Data Preprocessing
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "Data Preprocessing")
add_bullet_slide(slide, [
    "Image Resizing: All images resized to 224 x 224 pixels",
    "Normalization: ImageNet statistics (mean=[0.485, 0.456, 0.406])",
    "Data Augmentation (training only):",
    "    - Random Horizontal Flip (50%)",
    "    - Random Rotation (+/- 10 degrees)",
    "    - Color Jitter (brightness, contrast, saturation, hue)",
    "Class Imbalance: pos_weight = non_fresh_count / fresh_count",
    "Validation/Test: Only resize and normalize (no augmentation)"
], font_size=18)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 10: Methodology
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "Methodology")
steps = [
    ("Fish Image\nDataset", BLUE),
    ("Data\nPreprocessing", GREEN),
    ("Dataset\nSplitting", ORANGE),
    ("Model\nDevelopment", BLUE),
    ("Model\nTraining", GREEN),
    ("Model\nEvaluation", ORANGE),
    ("Prediction", BLUE)
]
for i, (step, clr) in enumerate(steps):
    x = 0.3 + i * 1.85
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(1.8), Inches(1.6), Inches(1.2))
    shape.fill.solid()
    shape.fill.fore_color.rgb = clr
    shape.line.fill.background()
    tf = shape.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = step
    p.font.size = Pt(14)
    p.font.color.rgb = WHITE
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER

add_bullet_slide(slide, [
    "The model learns visual patterns from training images",
    "During testing, it predicts freshness on unseen images",
    "Early stopping prevents overfitting",
    "LR scheduler adjusts learning rate when validation loss plateaus"
], top=4, font_size=20)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 11: Technology Used
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "Technology Used")
techs = [
    ("Python", "Programming Language"),
    ("PyTorch", "Deep Learning Framework"),
    ("EfficientNet-B0", "Pretrained CNN (ImageNet)"),
    ("torchvision", "Image Transforms & Datasets"),
    ("Pillow", "Image Loading"),
    ("NumPy", "Numerical Operations"),
    ("Matplotlib", "Graphs & Visualization"),
    ("scikit-learn", "Metrics & Confusion Matrix")
]
for i, (name, purpose) in enumerate(techs):
    row = i // 2
    col = i % 2
    x = 1 + col * 6
    y = 1.8 + row * 1.2
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(5), Inches(1))
    shape.fill.solid()
    shape.fill.fore_color.rgb = BLUE if col == 0 else GREEN
    shape.line.fill.background()
    tf = shape.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = f"{name}  —  {purpose}"
    p.font.size = Pt(18)
    p.font.color.rgb = WHITE
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 12: Model Architecture
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "Model Architecture — EfficientNet-B0")

arch_items = [
    ("Input Image\n224 x 224 x 3", BLUE),
    ("EfficientNet-B0\nBackbone (Frozen)", GREEN),
    ("Feature Vector\n1280-dimensional", ORANGE),
    ("Dropout\np = 0.3", BLUE),
    ("Linear Layer\n1280 → 1", GREEN),
    ("Sigmoid\nActivation", ORANGE),
    ("FRESH /\nNON-FRESH", BLUE)
]
for i, (item, clr) in enumerate(arch_items):
    x = 0.2 + i * 1.85
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(1.8), Inches(1.6), Inches(1.5))
    shape.fill.solid()
    shape.fill.fore_color.rgb = clr
    shape.line.fill.background()
    tf = shape.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = item
    p.font.size = Pt(13)
    p.font.color.rgb = WHITE
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER
    if i < len(arch_items) - 1:
        arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + 1.65), Inches(2.3), Inches(0.2), Inches(0.3))
        arrow.fill.solid()
        arrow.fill.fore_color.rgb = BLACK
        arrow.line.fill.background()

add_bullet_slide(slide, [
    "Why EfficientNet-B0? Best accuracy with fewer parameters",
    "Transfer Learning: Backbone frozen, only head is trained",
    "BCEWithLogitsLoss: Numerically stable binary classification",
    "Binary output: Single neuron (logit → sigmoid → probability)"
], top=4, font_size=18)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 13: Training Parameters
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "Training Parameters")

params = [
    ("Image Size", "224 x 224"),
    ("Batch Size", "32"),
    ("Epochs", "15"),
    ("Optimizer", "AdamW"),
    ("Learning Rate", "1e-3"),
    ("Weight Decay", "1e-2"),
    ("Loss Function", "BCEWithLogitsLoss"),
    ("LR Scheduler", "ReduceLROnPlateau"),
    ("Early Stopping", "Patience = 7"),
    ("Device", "MPS / CUDA / CPU")
]
for i, (param, value) in enumerate(params):
    row = i // 2
    col = i % 2
    x = 1 + col * 6
    y = 1.8 + row * 1
    add_text_box(slide, x, y, 2.5, 0.5, param, font_size=18, color=DARK_BLUE, bold=True)
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x + 2.8), Inches(y), Inches(2.5), Inches(0.5))
    shape.fill.solid()
    shape.fill.fore_color.rgb = BLUE
    shape.line.fill.background()
    tf = shape.text_frame
    p = tf.paragraphs[0]
    p.text = value
    p.font.size = Pt(16)
    p.font.color.rgb = WHITE
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 14: Project Structure
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "Project Structure")
files = [
    ("requirements.txt", "Dependencies"),
    ("utils.py", "Transforms, Dataloaders, Class Weights"),
    ("model.py", "EfficientNet-B0 Binary Classifier"),
    ("train.py", "Training Loop with Early Stopping"),
    ("evaluate.py", "Test Set Evaluation"),
    ("predict.py", "Single-Image Inference"),
    ("best_model.pth", "Trained Model Weights"),
    ("generate_results.py", "Graphs & Confusion Matrix")
]
for i, (fname, desc) in enumerate(files):
    row = i // 2
    col = i % 2
    x = 1 + col * 6
    y = 1.8 + row * 1.2
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(5.5), Inches(1))
    shape.fill.solid()
    shape.fill.fore_color.rgb = BLUE if col == 0 else GREEN
    shape.line.fill.background()
    tf = shape.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = f"{fname}  —  {desc}"
    p.font.size = Pt(16)
    p.font.color.rgb = WHITE
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 15: Training Results
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "Training Results")
add_text_box(slide, 1, 1.5, 11, 0.6, "Epoch-by-Epoch Progress", font_size=22, color=DARK_BLUE, bold=True)

epochs_data = [
    ("1", "67.40%", "86.29%", "40.77%", "98.57%"),
    ("3", "24.48%", "98.29%", "11.66%", "99.84%"),
    ("5", "15.95%", "98.57%", "6.30%", "100.00%"),
    ("7", "13.17%", "98.25%", "4.14%", "100.00%"),
    ("10", "8.43%", "99.20%", "2.66%", "100.00%"),
    ("15", "6.55%", "99.48%", "1.39%", "100.00%"),
]
headers = ["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc"]
for j, h in enumerate(headers):
    add_text_box(slide, 1 + j * 2.3, 2.2, 2, 0.5, h, font_size=16, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1 + j * 2.3), Inches(2.2), Inches(2), Inches(0.5))
    shape.fill.solid()
    shape.fill.fore_color.rgb = BLUE
    shape.line.fill.background()
    tf = shape.text_frame
    p = tf.paragraphs[0]
    p.text = h
    p.font.size = Pt(14)
    p.font.color.rgb = WHITE
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER

for i, row_data in enumerate(epochs_data):
    y = 2.8 + i * 0.5
    bg = LIGHT_GRAY if i % 2 == 0 else WHITE
    for j, val in enumerate(row_data):
        shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1 + j * 2.3), Inches(y), Inches(2), Inches(0.5))
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg
        shape.line.fill.background()
        tf = shape.text_frame
        p = tf.paragraphs[0]
        p.text = val
        p.font.size = Pt(14)
        p.font.color.rgb = BLACK
        p.alignment = PP_ALIGN.CENTER

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 16: Accuracy Graph
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "Accuracy Graph")
add_image_safe(slide, os.path.join(RESULTS_DIR, 'accuracy_graph.png'), 1.5, 1.5, width=10)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 17: Loss Graph
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "Loss Graph")
add_image_safe(slide, os.path.join(RESULTS_DIR, 'loss_graph.png'), 1.5, 1.5, width=10)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 18: Confusion Matrix
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "Confusion Matrix")
add_image_safe(slide, os.path.join(RESULTS_DIR, 'confusion_matrix.png'), 2, 1.5, width=9)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 19: Test Results
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "Test Results")

metrics = [
    ("Test Accuracy", "99.62%", GREEN),
    ("Precision (Fresh)", "99%", BLUE),
    ("Recall (Fresh)", "100%", BLUE),
    ("F1-Score (Fresh)", "99%", BLUE),
    ("Precision (Non-Fresh)", "100%", GREEN),
    ("Recall (Non-Fresh)", "99%", GREEN),
    ("F1-Score (Non-Fresh)", "100%", GREEN),
    ("Total Test Samples", "787", ORANGE),
]
for i, (name, value, clr) in enumerate(metrics):
    row = i // 2
    col = i % 2
    x = 1 + col * 6
    y = 1.8 + row * 1.2
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(5.5), Inches(1))
    shape.fill.solid()
    shape.fill.fore_color.rgb = clr
    shape.line.fill.background()
    tf = shape.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = f"{name}:  {value}"
    p.font.size = Pt(20)
    p.font.color.rgb = WHITE
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 20: Prediction Demo
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "Prediction Demo")
add_bullet_slide(slide, [
    "Command: python3 predict.py --image '/path/to/fish.jpg'",
    "",
    "Example 1 — Fresh Fish:",
    "    Image: IMG_2610.JPG",
    "    Prediction: FRESH",
    "    Confidence: 87.89%",
    "",
    "Example 2 — Non-Fresh Fish:",
    "    Image: IMG_7052.JPG",
    "    Prediction: NON-FRESH",
    "    Confidence: 99.29%",
    "",
    "The model outputs: Image name, Prediction, Confidence,",
    "Fresh probability, and Non-fresh probability"
], font_size=18)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 21: Advantages
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "Advantages")
add_bullet_slide(slide, [
    "Simple and easy-to-use system",
    "Fast image-based classification",
    "Reduces dependence on manual inspection",
    "Uses Transfer Learning — good results with limited data",
    "Handles class imbalance automatically",
    "Works on CPU, GPU, and Apple Silicon (MPS)",
    "Provides confidence scores for predictions",
    "99.62% accuracy on test data",
    "Can be improved by adding more training data"
], font_size=20)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 22: Limitations
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "Limitations")
add_bullet_slide(slide, [
    "Performance depends on dataset quality and diversity",
    "Different lighting conditions may affect prediction",
    "May not work equally well for all fish species",
    "Images with unusual angles may reduce accuracy",
    "Image-based prediction alone cannot confirm food safety",
    "Currently limited to binary classification (Fresh vs Non-Fresh)",
    "Trained primarily on fish eyes and gills close-up images"
], font_size=20)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 23: Future Scope
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "Future Scope")
add_bullet_slide(slide, [
    "Add more fish species to training data",
    "Increase dataset size and diversity",
    "Develop a mobile application for on-the-go detection",
    "Add real-time camera-based detection",
    "Multi-class classification (Fresh, Semi-Fresh, Spoiled)",
    "Fine-tune backbone layers for better feature extraction",
    "Deploy as a web application using Flask or FastAPI",
    "Add image quality assessment before prediction"
], font_size=20)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 24: Conclusion
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "Conclusion")
add_bullet_slide(slide, [
    "Built a Fish Freshness Detection system using Deep Learning",
    "Used EfficientNet-B0 with Transfer Learning",
    "Achieved 99.62% accuracy on 787 test images",
    "Only 3 misclassifications out of 787 images",
    "Handles class imbalance, augmentation, early stopping",
    "Provides automated, fast, and consistent predictions",
    "Demonstrates practical application of AI in food quality",
    "Can be extended to mobile/web deployment"
], font_size=20)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 25: References
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "References")
add_bullet_slide(slide, [
    "1. JIS Fish Classification Dataset — Kaggle",
    "     https://www.kaggle.com/datasets",
    "",
    "2. PyTorch Documentation",
    "     https://pytorch.org/docs/stable/",
    "",
    "3. EfficientNet: Rethinking Model Scaling for CNNs — Tan & Le, 2019",
    "",
    "4. EfficientNet-PyTorch",
    "     https://github.com/lukemelas/EfficientNet-PyTorch",
    "",
    "5. Torchvision Documentation",
    "     https://pytorch.org/vision/stable/"
], font_size=16)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 26: Thank You
# ══════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, DARK_BLUE)
add_text_box(slide, 1, 2.5, 11, 1.5, "THANK YOU", font_size=60, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
add_text_box(slide, 1, 4.2, 11, 0.8, "Questions?", font_size=32, color=ORANGE, bold=True, alignment=PP_ALIGN.CENTER)

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
output_path = '/Users/pranathi_tr/Desktop/ai ml/FishFreshAI/FishFreshAI_Presentation.pptx'
prs.save(output_path)
print(f'PPT saved to: {output_path}')
print(f'Total slides: {len(prs.slides)}')
