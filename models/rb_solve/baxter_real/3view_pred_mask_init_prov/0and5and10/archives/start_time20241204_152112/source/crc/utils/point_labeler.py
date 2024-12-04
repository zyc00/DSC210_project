import matplotlib.pyplot as plt
import imageio
import subprocess

import numpy as np
import cv2


class PointLabeler(object):
    def __init__(self, window_name="Point Labeler", screen_scale=1.0):
        self.window_name = window_name  # Name for our window
        self.done = False  # Flag signalling we're done
        self.current = (0, 0)  # Current position, so we can draw the line-in-progress
        self.points = np.empty((0, 2))  # List of points defining our polygon
        # self.labels = np.empty([0], dtype=int)
        self.screen_scale = screen_scale

    def on_mouse(self, event, x, y, flags, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if self.done:  # Nothing more to do
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                label = 0
            else:
                label = 1
            print(f"Adding point #{len(self.points)} with position({x},{y}), label {label}")

            if label == 0:
                x = y = -999
            self.points = np.concatenate((self.points, np.array([[x, y]])), axis=0)
            # self.labels = np.concatenate((self.labels, np.array([label])), axis=0)

            # self.detect()

    # def detect(self):
    #     input_point = self.points / self.ratio
    #     input_label = self.labels.astype(int)
    #
    #     masks, scores, logits = self.predictor.predict(
    #         point_coords=input_point,
    #         point_labels=input_label,
    #         multimask_output=True,
    #         return_logits=True,
    #     )
    #     maxidx = np.argmax(scores)
    #     mask = masks[maxidx]
    #     score = scores[maxidx]
    #     self.mask = mask.copy()
    #     # print()
    #     # for i, (mask, score) in enumerate(zip(masks, scores)):
    #     #     plt.figure(figsize=(10, 10))
    #     #     plt.imshow(image)
    #     #     show_mask(mask, plt.gca())
    #     #     show_points(input_point, input_label, plt.gca())
    #     #     plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
    #     #     plt.axis('off')
    #     #     plt.show()

    def run(self, rgb):
        image_to_show = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        output = subprocess.check_output(["xrandr"]).decode("utf-8")
        current_mode = [line for line in output.splitlines() if "*" in line][0]
        screen_width, screen_height = [int(x) for x in current_mode.split()[0].split("x")]
        scale = self.screen_scale
        screen_w = int(screen_width / scale)
        screen_h = int(screen_height / scale)

        image_h, image_w = image_to_show.shape[:2]
        ratio = min(screen_w / image_w, screen_h / image_h)
        # self.ratio = ratio
        # self.ratio = 1
        target_size = (int(image_w * ratio), int(image_h * ratio))
        image_to_show = cv2.resize(image_to_show, target_size)

        cv2.namedWindow(self.window_name)
        cv2.imshow(self.window_name, image_to_show)
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        while not self.done:
            tmp = image_to_show.copy()
            tmp = cv2.circle(tmp, self.current, radius=2,
                             color=(0, 0, 255),
                             thickness=-1)
            if self.points.shape[0] > 0:
                for ptidx, pt in enumerate(self.points):
                    color = (0, 255, 0)
                    tmp = cv2.circle(tmp, (int(pt[0]), int(pt[1])), radius=5,
                                     color=color,
                                     thickness=-1)
            cv2.imshow(self.window_name, tmp)
            if cv2.waitKey(50) == 27:  # ESC hit
                self.done = True
        cv2.destroyWindow(self.window_name)
        return self.points / ratio


def main():
    img_path = "tmp/dream_color.png"
    rgb = imageio.imread_v2(img_path)
    pointdrawer = PointLabeler(screen_scale=2.0)
    points = pointdrawer.run(rgb)
    print(points)

    #     test
    plt.imshow(rgb)
    for i in range(points.shape[0]):
        if points[i, 0] > 0:
            plt.scatter(points[i, 0], points[i, 1], c='r')
            plt.text(points[i, 0], points[i, 1], str(i))
    plt.show()


if __name__ == '__main__':
    main()
