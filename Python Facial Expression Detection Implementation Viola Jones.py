import cv2
import matplotlib.pyplot as plt

class FacialExpressionDetector:
    def __init__(self):
        # Load pre-trained Haar cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.nose_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_nose.xml')
        self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
    def detect_features(self, image_path, show_steps=False):
        """Detect facial features using Viola-Jones algorithm"""
        # Step 1: Read input image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image not found or unable to load")
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if show_steps:
            self._display_image("Step 1: Input Image", img)
        
        # Step 2: Face detection
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            print("No faces detected")
            return None
            
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            
            if show_steps:
                self._display_image("Step 2: Face Detection", img)
            
            # Step 3: Nose detection
            noses = self.nose_cascade.detectMultiScale(roi_gray, 1.7, 11)
            for (nx, ny, nw, nh) in noses:
                cv2.rectangle(roi_color, (nx, ny), (nx+nw, ny+nh), (0, 255, 0), 2)
                
            if show_steps and len(noses) > 0:
                self._display_image("Step 3: Nose Detection", img)
            
            # Step 4: Mouth detection
            mouths = self.mouth_cascade.detectMultiScale(roi_gray, 1.7, 20)
            for (mx, my, mw, mh) in mouths:
                cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (0, 0, 255), 2)
                
            if show_steps and len(mouths) > 0:
                self._display_image("Step 4: Mouth Detection", img)
            
            # Step 5: Eye detection
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 255, 0), 2)
                
            if show_steps and len(eyes) > 0:
                self._display_image("Step 5: Eye Detection", img)
        
        return img
    
    def _display_image(self, title, img):
        """Helper function to display images with titles"""
        plt.figure()
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
        plt.show()

# Example usage
if __name__ == "__main__":
    detector = FacialExpressionDetector()
    
    # Replace with your image path
    image_path = "test_image.jpg"
    
    try:
        result = detector.detect_features(image_path, show_steps=True)
        
        # Display final result
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title("Final Facial Feature Detection")
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")