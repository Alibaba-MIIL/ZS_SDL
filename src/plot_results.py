import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_results(img, pred_classes, classes, unseen_classes, path_output, name):
    height, width = img.size
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Create a black image
    pred_label_img = 255 * np.ones((height, 112, 3), np.uint8)
    step_size = 20
    starting_point = (10, 10)
    color = (0, 0, 255)
    for i, p in enumerate(pred_classes):
        if p in unseen_classes:
            text = "*" + p + "*"    # Unseen tags
        else:
            text = p
        current_point = (starting_point[0], starting_point[1] + step_size * i)
        pred_label_img = cv2.putText(pred_label_img, text, current_point, font, 0.5, color, 1)
    img_result = cv2.hconcat([np.array(img), pred_label_img])

    # Displaying image
    print('Saving image...')
    plt.figure()
    plt.imshow(img_result)
    plt.axis('off')
    plt.axis('tight')
    plt.title("Predicted tags")
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    plt.savefig(os.path.join(path_output, name))

    return img_result
