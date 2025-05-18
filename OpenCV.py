import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)  # Fill polygon with white
    return cv2.bitwise_and(img, mask)

def draw_lines(img, lines):
    if lines is None:
        return
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 5)

def process_frame(frame):
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    roi_vertices = np.array([[
        (int(0.1 * width), height),
        (int(0.45 * width), int(0.6 * height)),
        (int(0.55 * width), int(0.6 * height)),
        (int(0.9 * width), height)
    ]], dtype=np.int32)

    cropped_edges = region_of_interest(edges, roi_vertices)
    lines = cv2.HoughLinesP(
        cropped_edges,
        rho=2,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=40,
        maxLineGap=100
    )

    line_img = np.zeros_like(frame)
    draw_lines(line_img, lines)

    combo = cv2.addWeighted(frame, 0.8, line_img, 1, 1)
    return combo

def run_on_image(image_path):
    img = cv2.imread(image_path)
    result = process_frame(img)
    cv2.imshow('Lane Detection on Image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_on_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = process_frame(frame)
        cv2.imshow('Lane Detection on Video', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_on_image('road.png')
    run_on_video('road_video.mp4')