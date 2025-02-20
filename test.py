from thundersvm import SVC

# Create a dummy classifier to check GPU availability
try:
    clf = SVC(kernel='linear', gpu_id=0)  # Try using GPU 0
    print("ThunderSVM GPU support is available.")
except Exception as e:
    print("ThunderSVM GPU support is NOT available.")
    print(f"Error details: {e}")
