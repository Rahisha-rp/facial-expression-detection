# Facial Expression Detection using Viola-Jones Algorithm

![Sample Detection](images/sample_output.png)

This Python implementation detects facial features (face, eyes, nose, mouth) using the Viola-Jones algorithm (Haar cascades), based on the research paper by Rahisha Pokharel and Dr. Mandeep Kaur.

## Features
- Face detection
- Eye detection
- Nose detection
- Mouth detection
- Step-by-step visualization

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR-USERNAME/facial-expression-detection.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
```python
from facial_expression_detection import FacialExpressionDetector

detector = FacialExpressionDetector()
result = detector.detect_features("images/test_image.jpg", show_steps=True)
```

