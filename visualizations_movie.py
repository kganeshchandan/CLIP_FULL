import cv2
import os

image_folder = 'lineartraverse_imgs'
video_name = 'movie.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort()
print(images)

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()