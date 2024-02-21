import os
import numpy as np
import cv2

def get_xyhw(face_pts:np):
    min_x, min_y = min(face_pts[:, 0]), min(face_pts[:, 1])
    max_x, max_y = max(face_pts[:, 0]), max(face_pts[:, 1])

    # margin = 0.13
    # min_x = int(min_x * (1-margin))
    # min_y = int(min_y * (1-margin))
    # max_x = int(max_x * (1+margin))
    # max_y = int(max_y * (1+margin))
    height = max_y-min_y
    width = max_x-min_x
    return int(min_x), int(min_y), int(width), int(height)

def landmark_to_response(landmark:np, init=False) -> dict:
    '''
    랜드마크 반환 시, 얼굴 부위별 반환
    '''
    landmark_info = {'jaw':[0, 17], 'eyebrow_l':[17,22], 'eyebrow_r':[22,27], 'nose':[27,36], \
                'eye_left':[36,42], 'eye_right':[42,48],\
                'mouth_out':[48,60], 'mouth_in':[60,68],}
    landmark = landmark.astype(int) #.tolist()

    face_landmark = {}
    for key, value in landmark_info.items():
        face_landmark[key] = dict()
        face_landmark[key]["landmark"] = landmark[value[0]:value[1]].tolist()
        # np.int64 반환되어 안되고 있던 문제로, int형으로 바꾸어 성공
        # json 반환 시, 안에 담긴 숫자의 data type이 np.int64일 경우 반환 불가 -> int형으로 감싸 반환해야 함.
        if init:
            face_landmark[key]["bbox"] = get_xyhw(landmark[value[0]:value[1]])
    return face_landmark

def draw_bbox(img, face_landmark_bbox, pc=(0,0,255), radius=2, lc=(0,255,0), thickness=2):
    print("draw_bbox")
    landmark_info = {'jaw':[0, 17], 'eyebrow_l':[17,22], 'eyebrow_r':[22,27], 'nose':[27,36], \
                'eye_left':[36,42], 'eye_right':[42,48],\
                'mouth_out':[48,60], 'mouth_in':[60,68],}
    for key, value in landmark_info.items():
        x,y,height,width, = face_landmark_bbox[key]['bbox']
        cv2.rectangle(img, (x, y), (x + height, y + width),
                (0, 0, 255), thickness=2)
    return img


def closest_node(xy, pts):
    #search the list of nodes for the one closest to node, return the name
    dist_2 = np.sqrt(np.sum((pts - np.array(xy).reshape((-1, 2)))**2, axis=1))
    if (dist_2[np.argmin(dist_2)] > 20):
        return -1
    return np.argmin(dist_2)

## input ##
img0 = cv2.imread("sample/prev-0-w.png")
img = np.copy(img0)
pts = np.loadtxt("sample/prev-0_face_open_mouth.txt")
face_landmark_bbox = landmark_to_response(pts, init=True) # 얼굴 부위별로 bbox 관리 추가 
node = -1

img = draw_bbox(img, face_landmark_bbox)

def click_adjust_wireframe(event, x, y, flags, param):
    global img, pts, node, face_landmark_bbox

    def update_img(node, button_up=False):
        global img, pts, face_landmark_bbox

        # update carton points object and get fresh pts list
        pts[node, 0], pts[node, 1] = x, y


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
        # 왼쪽 마우스 버튼 눌렀을 때 좌표 가져오기
        node = closest_node((x, y), pts)
        if(node >=0):
            update_img(node)

    if event == cv2.EVENT_LBUTTONUP:
        # 왼쪽 마우스 버튼을 뗐을 때 좌표 가져오기
        node = closest_node((x, y), pts)
        if (node >= 0):
            update_img(node, button_up=True)
        node = -1

    if event == cv2.EVENT_MOUSEMOVE:
        # redraw figure
        if (node != -1):
            update_img(node)
            


cv2.namedWindow("frontend", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("frontend", click_adjust_wireframe)

while(True):
    cv2.imshow('frontend', img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cv2.destroyAllWindows()
