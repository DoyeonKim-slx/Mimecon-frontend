import os
import numpy as np
import cv2

def cal_xy12(face_pts:np):
    min_x, min_y = min(face_pts[:, 0]), min(face_pts[:, 1])
    max_x, max_y = max(face_pts[:, 0]), max(face_pts[:, 1])

    height = max_y-min_y
    width = max_x-min_x
    return [int(min_x), int(min_y)], [int(max_x), int(max_y)]

def get_bbox(landmark:np) -> dict:
    '''
    랜드마크 반환 시, 얼굴 부위별 반환
    '''
    landmark_info = {'jaw':[0, 17], 'eyebrow_l':[17,22], 'eyebrow_r':[22,27], 'nose':[27,36], \
                'eye_left':[36,42], 'eye_right':[42,48],\
                'mouth_out':[48,60], 'mouth_in':[60,68],}
    landmark = landmark.astype(int) #.tolist()

    bbox_list = []
    for key, value in landmark_info.items():
        xy12 = cal_xy12(landmark[value[0]:value[1]])
        bbox_list.append(xy12[0])
        bbox_list.append(xy12[1])
    return bbox_list 

def draw_bbox(img, bbox_info_list, pc=(0,0,255), radius=5, lc=(0,255,0), thickness=2):
    # landmark_info = {'jaw':[0, 17], 'eyebrow_l':[17,22], 'eyebrow_r':[22,27], 'nose':[27,36], \
    #             'eye_left':[36,42], 'eye_right':[42,48],\
    #             'mouth_out':[48,60], 'mouth_in':[60,68],}
    
    for idx in range(0, len(bbox_info_list), 2):
        x1,y1, = bbox_info_list[idx]
        x2,y2, = bbox_info_list[idx+1]
        cv2.rectangle(img, (x1, y1), (x2,y2), pc, thickness=2)
        cv2.circle(img, (x1, y1), radius, pc, -1)
        cv2.circle(img, (x2,y2), radius, pc, -1)
    return img

def closest_node(xy, bbox_info_list):
    #search the list of nodes for the one closest to node, return the name
    dist_2 = np.sqrt(np.sum((bbox_info_list - np.array(xy).reshape((-1, 2)))**2, axis=1))
    if (dist_2[np.argmin(dist_2)] > 20):
        return -1
    return np.argmin(dist_2)

def closest_coordinate(xy, bbox_info_list):
    min_distance = float('inf')
    closest_coord = None
    closest_idx = None

    for idx, coord in enumerate(bbox_info_list):
        distance = np.linalg.norm(np.array(coord) - np.array(xy))
        if distance < min_distance:
            min_distance = distance
            closest_coord = coord
            closest_idx = idx
    

    return closest_idx

## input ##
img0 = cv2.imread("sample/prev-0-w.png")
img = np.copy(img0)
pts = np.loadtxt("sample/prev-0_face_open_mouth.txt")
bbox_info_list = get_bbox(pts) 
print(bbox_info_list)
node = -1

img = draw_bbox(img, bbox_info_list)

def click_adjust_wireframe(event, x, y, flags, param):
    global img, pts, node, bbox_info_list

    def update_img(node, button_up=False):
        global img, pts, bbox_info_list

        # update carton points object and get fresh pts list
        # xy 좌표를 받아와 해당 좌표를 pts 변수에 업데이트
        # bbox의 좌표를 받아와 업데이트 해서 결과 이미지에 그림 그려주면 됨. 
        bbox_info_list[node][0], bbox_info_list[node][1] = x, y

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
        print("EVENT_LBUTTONDOWN")
        # search for nearest point
        # 왼쪽 마우스 버튼 눌렀을 때 가까운 node 가져오기
        # x, y는 EVENT_LBUTTONDOWN의 반환 값
        node = closest_coordinate((x, y), bbox_info_list)
        if(node >=0):
            update_img(node)

    if event == cv2.EVENT_LBUTTONUP:
        # 왼쪽 마우스 버튼을 뗐을 때 좌표 가져오기
        node = closest_coordinate((x, y), bbox_info_list)
        if (node >= 0):
            update_img(node, button_up=True)
        node = -1

    if event == cv2.EVENT_MOUSEMOVE:
        # redraw figure
        if (node != -1):
            update_img(node)
            
# draw_bbox(img, bbox_info_list)

cv2.namedWindow("frontend", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("frontend", click_adjust_wireframe)

while(True):
    cv2.imshow('frontend', img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cv2.destroyAllWindows()
