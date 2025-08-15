# 1. Start from an official Python base image
FROM python:3.11-slim

# 2. Install a comprehensive set of OS-level dependencies for OpenCV
# This includes libraries for OpenGL, GLib, X11, and video processing (ffmpeg)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    ffmpeg \
    && apt-get clean

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Copy the requirements file into the container
COPY requirements.txt .

# 5. Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy your entire project code into the container
COPY . .

# 7. Expose the port the app runs on
EXPOSE 8000

# 8. Define the command to run your application
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"] 