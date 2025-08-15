Real-Time Warehouse Safety Monitoring System
A deep learning application built with YOLOv8 to detect personal protective equipment (PPE) violations in real-time, enhancing workplace safety through automated monitoring and instant audio-visual alerts.

This project demonstrates an end-to-end computer vision pipeline, from model training to deployment in a live application. It uses a state-of-the-art object detection model to identify persons, helmets, and safety vests from a live camera feed, determines if a safety violation has occurred, and provides immediate feedback.

🎥 Project Demo
(This is where you would embed a GIF or a short video of your application in action. It's the most effective way to showcase your work.)

✨ Key Features
Real-Time Object Detection: Utilizes the YOLOv8 model to accurately detect multiple classes (person, head, helmet, vest) in real-time from a video stream.

Safety Violation Logic: Intelligently determines if a safety violation is occurring by checking if a person or head is detected without a corresponding helmet.

Interactive Visual Dashboard: A clean, user-friendly interface built with OpenCV that displays the live camera feed, bounding boxes, and a clear status panel.

Instant Audio Alerts: Provides immediate, audible warnings using Google's Text-to-Speech (gTTS) engine the moment a safety violation is detected or cleared, ensuring immediate attention.

Modular and Organized Codebase: Follows professional software engineering practices with a structured project directory, making the code easy to understand, maintain, and extend.

🛠️ Tech Stack & Concepts
This project leverages a modern stack of AI and software development tools:

Programming Language: Python 3.12

Core AI/ML Frameworks:

PyTorch: The underlying deep learning framework for model training and inference.

Ultralytics YOLOv8: A state-of-the-art, real-time object detection model used for its high speed and accuracy.

Computer Vision:

OpenCV: Used for all video stream capture, image processing, and for drawing the visual dashboard and bounding boxes.

Audio Feedback:

gTTS (Google Text-to-Speech): Used to generate high-quality, natural-sounding audio alerts.

Playsound: A cross-platform library used to play the generated audio files.

Environment Management:

venv: Used to create an isolated Python environment to manage project dependencies.

pip: Python's package installer.

📂 Project Structure
The project is organized into a standard machine learning directory structure for clarity and scalability.

warehouse_vision/
│
├── data/
│   ├── train/          # Training images and labels
│   ├── valid/          # Validation images and labels
│   └── test/           # Test images and labels
│   └── data.yaml       # Dataset configuration file for YOLO
│
├── models/
│   └── hard_hat_detection_final4/ # Folder where trained models are saved
│       └── weights/
│           └── best.pt     # The best performing trained model weights
│
├── src/
│   ├── __init__.py
│   ├── train.py        # Script to train the YOLOv8 model
│   └── predict.py      # Script to run the live detection application
│
├── .gitignore          # Specifies files for Git to ignore
└── README.md           # You are here!

🚀 Setup and Installation
Follow these steps to set up and run the project on your local machine.

1. Clone the Repository
git clone <your-repository-url>
cd warehouse_vision

2. Create and Activate a Virtual Environment
It's highly recommended to use a virtual environment.

# Create the environment
python -m venv venv

# Activate it (macOS/Linux)
source venv/bin/activate

# Activate it (Windows)
.\venv\Scripts\activate

3. Install Dependencies
This project uses specific library versions for compatibility. A requirements.txt file should be created for easy installation.

# First, create the requirements.txt file with these contents:
# ultralytics
# opencv-python
# gTTS
# playsound==1.2.2

# Then, install from the file
pip install -r requirements.txt

⚙️ Usage
The project has two main scripts located in the src/ folder.

1. Train the Model
To train the model on the dataset, run the train.py script from the project root directory.

python src/train.py

This will start the training process and save the best model weights to the models/ directory.

2. Run the Live Monitoring Application
To run the live safety detection system using your webcam, execute the predict.py script.

python src/predict.py

A window will appear showing your camera feed with the real-time dashboard and alerts. Press 'q' to quit.

🧠 Challenges & Learnings
This project involved overcoming several common but critical software engineering challenges:

Dependency Conflicts: The initial setup faced a RuntimeError due to a version conflict between NumPy and PyTorch. This was resolved by forcing a clean reinstallation of ultralytics and its dependencies, ensuring a stable and compatible environment. This highlights the importance of precise dependency management in complex projects.

File Path Management: The program initially failed with a FileNotFoundError. This was traced back to incorrect relative paths in both the Python scripts and the data.yaml configuration file. The solution involved restructuring the data directory and using os.path.join to create robust, OS-agnostic paths, reinforcing the need for careful path handling.

Text-to-Speech Library Failure: The first TTS library (pyttsx3) failed to work on the target macOS system. Instead of getting stuck, the project pivoted to a more reliable, web-based library (gTTS), demonstrating adaptability and the ability to find alternative solutions when a primary tool fails.

🔮 Future Improvements
Object Tracking: Implement object tracking to assign a unique ID to each person, allowing for more granular monitoring.

Database Logging: Log all violation events to a database (like SQLite or PostgreSQL) with timestamps for auditing and analysis.

Web-Based Dashboard: Develop a web interface using Flask or FastAPI to allow remote monitoring from any device on the network.

Model Optimization: Convert the trained model to an optimized format like ONNX or TensorRT to achieve higher FPS on edge devices or CPUs.