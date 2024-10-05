import cv2
import os
from tqdm import tqdm

images_path = "../datasets/ExpW/image/origin"
save_path = "../datasets/ExpW_clean"
coords_path = "../datasets/ExpW/label/label.lst"

targets = [0, 1, 2, 3, 4, 5, 6]

# create directory
# if not os.path.isdir(save_path):
#     os.mkdir(save_path)

# create target directories
for target in targets:
    if not os.path.isdir(f"{save_path}/{target}"):
        os.mkdir(f"{save_path}/{target}")

counter = 0
filenames = []
with open(coords_path, "r") as file:
    for line in tqdm(file):
        if len(line) > 1:  # puste linie
            (
                file_name,
                face_id,
                face_box_top,
                face_box_left,
                face_box_right,
                face_box_bottom,
                confidence,
                expression_label,
            ) = line.split(" ")
            
            try:
                expression_label = expression_label[0]
                image = cv2.imread(f"{images_path}/{file_name}")
                img_new = image.copy()
                img_new = img_new[
                    int(face_box_top) : int(face_box_bottom),
                    int(face_box_left) : int(face_box_right),
                ]
                file_name = file_name.split(".")[0]
                cv2.imwrite(
                    f"{save_path}/{expression_label}/{file_name}_{face_id}.png", img_new
                )
            except:
                counter += 1
                filenames.append(file_name)

print(counter)
print(file_name)

