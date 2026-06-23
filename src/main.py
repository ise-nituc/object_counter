from ultralytics import YOLO
from numpy import ndarray
import cv2


class ObjectCounter:
    
    """A class to count objects in real-time video using YOLOv8."""

    def __init__(self, model_path: str ="yolov8n.onnx") -> None:
        
        """
        Initialize the ObjectCounter.
        
        Args:
            model_path (str): Path to the YOLO model file.
        """
        
        self.model = YOLO(model_path)
        self.names = self.model.names
        self.target_class = None
        self.target_label = None
        self.cap = None
        self.window_name = "Object Counter"

        # Configuration
        self.target_resolution = (1920, 1080)
        self.target_fps = 30
        self.model_input_size = 416
        self.confidence_threshold = 0.25
        self.window_size = (800, 600)
        self.escape_key = 27

    def display_available_classes(self) -> None:
        
        """Print all available COCO object classes in a formatted table."""

        print("\n" + "=" * 50)
        print("COCO Object Classes:")
        print("=" * 50)
        for class_id, label in self.names.items():
            print(f"{class_id:3d} : {label}")
        print("=" * 50)

    def get_target_class_from_user(self) -> bool:

        """
        Prompt user to select a class to count.
        
        Returns:
            bool: True if valid class was selected, False otherwise.
        """

        while True:
            try:
                user_input = input("\nEnter the class ID to count (0-79): ")
                class_id = int(user_input)
                
                # Default to 0 (person) if user presses Enter without input
                if user_input == "":
                    class_id = 0
                elif 0 <= int(user_input) <= 79:
                    self.target_class = class_id
                    self.target_label = self.names[class_id]
                    print(f"\n✓ You will count: {self.target_label}\n")
                    break
                else:
                    print("Please enter a number between 0 and 79")
            except ValueError:
                print("Please enter a valid number")

    def initialize_camera(self):

        """Set up webcam with desired resolution and frame rate."""

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        # Create resizable window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_size[0], self.window_size[1])

        # Print camera info
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Capture resolution: {actual_width:.0f}x{actual_height:.0f}")
        print("Press ESC to exit\n")

    def process_frame(self, frame: ndarray) -> tuple[ndarray, int]:

        """
        Process a single frame and return predictions.
        
        Args:
            frame: Input frame from video capture
            
        Returns:
            tuple: (annotated_frame, detection_count)
        """

        # Resize frame to model input size
        frame_resized = cv2.resize(frame, (self.model_input_size, self.model_input_size))

        # Run inference
        results = self.model.predict(
            frame_resized,
            imgsz=self.model_input_size,
            conf=self.confidence_threshold,
            classes=[self.target_class],
            verbose=False
        )

        # Get annotated frame and detection count
        annotated = results[0].plot()
        count = len(results[0].boxes)

        return annotated, count

    def add_count_text(self, frame: ndarray, count: int) -> ndarray:

        """
        Add count text to the frame.
        
        Args:
            frame: Input frame
            count (int): Number of objects detected
            
        Returns:
            frame: Frame with text overlay
        """

        text = f"# {self.target_label}: {count}"
        cv2.putText(
            frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (150, 255, 0),
            2
        )

        return frame

    def run(self):

        """Main loop for object detection and counting."""

        self.display_available_classes()
        self.get_target_class_from_user()
        self.initialize_camera()

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break

                # Process frame
                annotated, count = self.process_frame(frame)

                # Add count overlay
                annotated = self.add_count_text(annotated, count)

                # Display frame
                cv2.imshow(self.window_name, annotated)

                # Check for exit key
                if cv2.waitKey(1) == self.escape_key:
                    break
        finally:
            self.cleanup()

    def cleanup(self):
        """Release resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("\nCamera released.")


if __name__ == "__main__":

    counter = ObjectCounter(model_path="yolov8n.onnx")
    counter.run()