import cv2

input_file = 'input.mp4'
output_file = 'short_input.mp4'

cap = cv2.VideoCapture(input_file)
fps = int(cap.get(cv2.CAP_PROP_FPS))

start_frame = 3 * 60 * fps
end_frame = start_frame + (15 * fps)

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

for _ in range(end_frame - start_frame):
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)

cap.release()
out.release()
print(f"Clip saved as {output_file}")
