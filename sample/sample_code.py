import os
import numpy as np
import cv2

# keypoint 
def draw_landmarks(img, pts, pc=(0,0,255), radius=2, lc=(0,255,0), thickness=2):

    for i in range(0, 16): # 16 jaw
        cv2.line(img, (int(pts[i, 0]), int(pts[i, 1])),
                     (int(pts[i+1, 0]), int(pts[i+1, 1])), (0, 255, 0), thickness)
    for i in range(17, 21): # 4 eyebrow
        cv2.line(img, (int(pts[i, 0]), int(pts[i, 1])),
                     (int(pts[i+1, 0]), int(pts[i+1, 1])), (255, 0, 0), thickness)
    for i in range(22, 26): # 4
        cv2.line(img, (int(pts[i, 0]), int(pts[i, 1])),
                     (int(pts[i+1, 0]), int(pts[i+1, 1])), (255, 0, 0), thickness)
    for i in range(27, 35): # 8 nose
        cv2.line(img, (int(pts[i, 0]), int(pts[i, 1])),
                     (int(pts[i+1, 0]), int(pts[i+1, 1])), (255, 255, 0), thickness)
    for i in range(36, 41): # 5 eye
        cv2.line(img, (int(pts[i, 0]), int(pts[i, 1])),
                     (int(pts[i+1, 0]), int(pts[i+1, 1])), (255, 0, 255), thickness)
    for i in range(42, 47): # 5
        cv2.line(img, (int(pts[i, 0]), int(pts[i, 1])),
                     (int(pts[i+1, 0]), int(pts[i+1, 1])), (255, 0, 255), thickness)
    for i in range(48, 59): # 11 mouse out
        cv2.line(img, (int(pts[i, 0]), int(pts[i, 1])),
                     (int(pts[i+1, 0]), int(pts[i+1, 1])), (255, 128, 0), thickness)
    for i in range(60, 67): # 7 mouse in
        cv2.line(img, (int(pts[i, 0]), int(pts[i, 1])),
                     (int(pts[i+1, 0]), int(pts[i+1, 1])), (255, 128, 128), thickness)

    for i in range(68):
        cv2.circle(img, (int(pts[i, 0]), int(pts[i, 1])), radius, pc, -1)

def closest_node(xy, pts):
    #search the list of nodes for the one closest to node, return the name
    dist_2 = np.sqrt(np.sum((pts - np.array(xy).reshape((-1, 2)))**2, axis=1))
    if (dist_2[np.argmin(dist_2)] > 20):
        return -1
    return np.argmin(dist_2)

# 사람이 랜드마크를 수정하는 부분
def click_adjust_wireframe(event, x, y, flags, param):
    global img, pts, node

    def update_img(node, button_up=False):
        global img, pts

        # update carton points object and get fresh pts list
        pts[node, 0], pts[node, 1] = x, y

        img = np.copy(img0)
        draw_landmarks(img, pts)

        # 랜드마크의 해당 좌표를 마우스로 클릭했을 때 확대 이미지가 함께 뜨도록 함.
        # zoom-in feature
        if (not button_up):
            zoom_in_scale = 2
            zoom_in_box_size = int(150 / zoom_in_scale)
            zoom_in_range = int(np.min([zoom_in_box_size, x, y,
                                        (img.shape[0] - y) / 2 / zoom_in_scale,
                                        (img.shape[1] - x) / 2 / zoom_in_scale]))

            img_zoom_in = img[y - zoom_in_range:y + zoom_in_range,
                          x - zoom_in_range:x + zoom_in_range].copy()
            img_zoom_in = cv2.resize(img_zoom_in, (0, 0), fx=zoom_in_scale,
                                     fy=zoom_in_scale)
            cv2.drawMarker(img_zoom_in, (zoom_in_range * zoom_in_scale,
                                         zoom_in_range * zoom_in_scale),
                           (0, 0, 255),
                           markerType=cv2.MARKER_CROSS, markerSize=30,
                           thickness=2, line_type=cv2.LINE_AA)
            height, width, depth = np.shape(img_zoom_in)

            img[y:y + height, x:x + width] = img_zoom_in
            cv2.rectangle(img, (x, y), (x + height, y + width),
                          (0, 0, 255), thickness=2)


    if event == cv2.EVENT_LBUTTONDOWN:
        # search for nearest point
        node = closest_node((x, y), pts)
        if(node >=0):
            update_img(node)

    if event == cv2.EVENT_LBUTTONUP:
        node = closest_node((x, y), pts)
        if (node >= 0):
            update_img(node, button_up=True)
        node = -1

    if event == cv2.EVENT_MOUSEMOVE:
        # redraw figure
        if (node != -1):
            update_img(node)

draw_landmarks(img, pts)

cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("img", click_adjust_wireframe)

while(True):
    cv2.imshow('img', img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cv2.destroyAllWindows()
