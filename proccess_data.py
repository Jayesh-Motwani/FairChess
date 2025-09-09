import cv2 as cv
import numpy as np
import os


# Function 1: This function remains the same.
def find_chessboard_corners(img):
    """
    Takes a loaded image, finds the four corners of the largest contour,
    and returns them.
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blur, 50, 150)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest_contour = max(contours, key=cv.contourArea)
    points = largest_contour.reshape(len(largest_contour), 2)
    corners = np.zeros((4, 2), dtype="float32")

    s = points.sum(axis=1)
    corners[0] = points[np.argmin(s)]
    corners[2] = points[np.argmax(s)]
    diff = np.diff(points, axis=1)
    corners[1] = points[np.argmin(diff)]
    corners[3] = points[np.argmax(diff)]

    return corners


# Function 2: Modified to accept and return image counts
def create_dataset_from_corners(original_image, corners, img_counts, output_dir='dataset'):
    """
    Warps, crops, and saves squares, using a running count of images.
    """
    dst_points = np.array([[0, 0], [799, 0], [799, 799], [0, 799]], dtype="float32")
    M = cv.getPerspectiveTransform(corners, dst_points)
    warped_board = cv.warpPerspective(original_image, M, (800, 800))

    piece_map = [
        ['br', 'bn', 'bb', 'bq', 'bk', 'bb', 'bn', 'br'],
        ['bp', 'bp', 'bp', 'bp', 'bp', 'bp', 'bp', 'bp'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e', 'e'], ['e', 'e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e', 'e'], ['e', 'e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['wp', 'wp', 'wp', 'wp', 'wp', 'wp', 'wp', 'wp'],
        ['wr', 'wn', 'wb', 'wq', 'wk', 'wb', 'wn', 'wr']
    ]

    square_size = 100
    for row in range(8):
        for col in range(8):
            square_img = warped_board[row * square_size:(row + 1) * square_size,
                         col * square_size:(col + 1) * square_size]
            piece_label = piece_map[row][col]
            piece_dir = os.path.join(output_dir, piece_label)
            os.makedirs(piece_dir, exist_ok=True)

            count = img_counts.get(piece_label, 0)
            filename = f"{piece_label}_{count}.png"
            filepath = os.path.join(piece_dir, filename)

            cv.imwrite(filepath, square_img)
            img_counts[piece_label] = count + 1

    # Return the updated counts for the next image
    return img_counts


# --- Main Script with Automation Loop ---
if __name__ == '__main__':
    RAW_DATA_DIR = 'Raw Data'
    OUTPUT_DIR = 'dataset'

    # Dictionary to hold the running total of images saved for each piece
    image_counts = {}

    # Check if the raw data directory exists
    if not os.path.exists(RAW_DATA_DIR):
        print(f"Error: Directory '{RAW_DATA_DIR}' not found. Please create it and add your images.")
    else:
        # Loop over every file in the raw_data directory
        for filename in os.listdir(RAW_DATA_DIR):
            # Check if the file is an image
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(RAW_DATA_DIR, filename)
                print(f"Processing {image_path}...")

                image = cv.imread(image_path)
                if image is None:
                    print(f"  - Could not read image, skipping.")
                    continue

                # Step 1: Find the corners
                board_corners = find_chessboard_corners(image)

                # Step 2: If corners are found, create the dataset
                if board_corners is not None:
                    # Pass the current counts and get the updated counts back
                    image_counts = create_dataset_from_corners(image, board_corners, image_counts, OUTPUT_DIR)
                    print(f"  - Successfully processed and saved 64 squares.")
                else:
                    print(f"  - Could not find chessboard corners, skipping.")

        print("\nAutomation complete! Dataset created in the 'dataset' folder.")