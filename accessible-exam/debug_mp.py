import mediapipe as mp
print(f"Mediapipe file: {mp.__file__}")
try:
    print(f"Solutions: {mp.solutions}")
except AttributeError:
    print("No solutions attribute found")
