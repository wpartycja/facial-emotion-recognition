import cv2

images_path = 'test_images/expw'
save_path = 'faces/expw/'

with open('coords.txt', 'r') as file:
    for line in file:
        if len(line) > 1:   # puste linie
            file_name, face_id, face_box_top, face_box_left, face_box_right, face_box_bottom, confidence, expression_label = line.split(' ')
            image = cv2.imread(f'{images_path}/{file_name}')
            img_new = image.copy()
            img_new = img_new[int(face_box_top):int(face_box_bottom), int(face_box_left):int(face_box_right)]
            file_name = file_name.split('.')[0]
            with open(f'{save_path}{file_name}.png', 'w') as fp:
                pass
            cv2.imwrite(f'{save_path}{file_name}.png', img_new)