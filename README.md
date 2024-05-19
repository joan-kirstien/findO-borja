# How to run FindO

## Prerequisites

Before running the voice assistant, ensure you have the following prerequisites installed:

- Install Python 3.8 or newer.
- Install the necessary Python packages:
  ```bash
  pip install ultralytics opencv-python-headless numpy pytorch torchvision torchaudio SpeechRecognition pyttsx3 better-profanity textblob nltk
  ```
- Clone the repository and navigate to the directory where the `va.py` script is located.

### Instructions

1. **Run the Voice Assistant**: Execute the `va.py` script by running the following command in your terminal:
   ```bash
   python va.py
   ```

2. **Interacting with the Voice Assistant**:
   - After starting the voice assistant, it will greet you with: "Hello What can I do for you today?"
   - You can interact with the voice assistant by speaking one of the following commands:
     - "hello"
     - "how are you"
     - "what's your name"
     - "locate" (This command activates the object detection feature)
     - "bye" (This command ends the session)

### Note

- The voice assistant uses Google's speech recognition service, so ensure you have a stable internet connection.
- The object detection feature requires a webcam connected to your device. Make sure the webcam is functioning correctly.

---

This README.md content provides clear instructions for setting up the environment, running the voice assistant, and how to interact with it. It also includes notes about dependencies and hardware requirements.

Citations:
