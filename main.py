import os
import numpy as np
import cv2
from dataclasses import dataclass

from src import selectinwindow

@dataclass
class Rect:
    x:int = 0
    y:int = 0
    w:int = 0
    h:int = 0

    def is_inside(self, px:int, py:int):
        return (self.x <= px <= self.x+w) and (self.y <= py <= self.y+h)
    
    def copy(self):
        return Rect(self.x,self.y,self.w,self.h)


class AppWindow:
    marker_size:int = 8
    
    drag = False
    TL = False
    TM = False
    TR = False
    LM = False
    RM = False
    BL = False
    BM = False
    BR = False

    hold = False
    
    def __init__(self, img, window_name:str, limits:Rect, bbox:Rect):
        self.image = img
        self.window_name = window_name

        self.limits = limits
        self.bbox = bbox

    @staticmethod
    def drag_rect(event, x,y, flags, self):
        x = min(max(x,self.limits.x),self.limits.x+self.limits.w-1)
        y = min(max(y,self.limits.y),self.limits.y+self.limits.h-1)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_down(x, y)
        if event == cv2.EVENT_LBUTTONUP:
            self.mouse_up()
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_move(x, y)
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.mouse_double_click(x, y)

    def mouse_down(self, x,y):
        lx,mx,rx = self.bbox.x, self.bbox.x+self.bbox.w/2, self.bbox.x+self.bbox.w
        ty,my,by = self.bbox.y, self.bbox.y+self.bbox.h/2, self.bbox.y+self.bbox.h
        msh = self.marker_size/2
        if (ty-msh <= y <= ty+msh):
            if (lx-msh <= x <= lx+msh):
                self.TL = True
                return
            if (mx-msh <= x <= mx+msh):
                self.TM = True
                return
            if (rx-msh <= x <= rx+msh):
                self.TR = True
                return
        
        if (my-msh <= y <= my+msh):
            if (lx-msh <= x <= lx+msh):
                self.LM = True
                return
            if (rx-msh <= x <= rx+msh):
                self.RM = True
                return
            
        if (by-msh <= y <= by+msh):
            if (lx-msh <= x <= lx+msh):
                self.BL = True
                return
            if (mx-msh <= x <= mx+msh):
                self.BM = True
                return
            if (rx-msh <= x <= rx+msh):
                self.BR = True
                return
        
        if self.bbox.is_inside(x,y):
            self.anchor = self.bbox.copy()
            self.offset_x = x
            self.offset_y = y
            self.hold = True
            return
    
    def disable_resize_buttons(self):
        self.TL = self.TM = self.TR = False
        self.LM = self.RM = False
        self.BL = self.BM = self.BR = False
        self.hold = False
    
    def straighten_up_rect(self):
        if self.bbox.w < 0:
            self.bbox.x = self.bbox.x+self.bbox.w
            self.bbox.w = -self.bbox.w
        if self.bbox.h < 0:
            self.bbox.h = self.bbox.y+self.bbox.h
            self.bbox.h = -self.bbox.h
        
    def clear_canvas_n_draw(self):

        tmp = self.image.copy()
        cv2.rectangle(
            tmp,
            (self.bbox.x, self.bbox.y),
            (self.bbox.x+self.bbox.w, self.bbox.y+self.bbox.h),
            (0, 255, 0),
            2)
        for (x,y) in [
            (self.bbox.x, self.bbox.y),
            (self.bbox.x, self.bbox.y+self.bbox.h//2),
            (self.bbox.x, self.bbox.y+self.bbox.h),

            (self.bbox.x+self.bbox.w//2, self.bbox.y),
            (self.bbox.x+self.bbox.w//2, self.bbox.y+self.bbox.h),

            (self.bbox.x+self.bbox.w, self.bbox.y),
            (self.bbox.x+self.bbox.w, self.bbox.y+self.bbox.h//2),
            (self.bbox.x+self.bbox.w, self.bbox.y+self.bbox.h),
        ]:
            cv2.rectangle(
                tmp,
                (x-self.marker_size//2,y-self.marker_size//2),
                (x+self.marker_size//2,y+self.marker_size//2),
                (0, 255, 0),
                2)
        cv2.imshow(self.window_name, tmp)
        cv2.waitKey(0)

    def mouse_up(self):
        self.drag = False
        
        self.disable_resize_buttons()
        self.straighten_up_rect()
        self.clear_canvas_n_draw()

    def mouse_move(self, x,y):
        if self.hold:
            self.bbox.x = self.anchor.x + (x - self.offset_x)
            self.bbox.y = self.anchor.x + (y - self.offset_y)

            if self.bbox.x < self.limits.x:
                self.bbox.x = self.limits.x
            if self.bbox.x+self.bbox.w > self.limits.x+self.limits.w-1:
                self.bbox.x = self.bbox.x+self.limits.w-self.bbox.w-1
            if self.bbox.y < self.limits.y:
                self.bbox.y = self.limits.y
            if self.bbox.y+self.bbox.h > self.limits.y+self.limits.h-1:
                self.bbox.y = self.bbox.y+self.limits.h-self.bbox.h-1

            self.clear_canvas_n_draw()
            return
            
        if self.TL:
            self.bbox.w = (self.bbox.x+self.bbox.w)-x
            self.bbox.h = (self.bbox.y+self.bbox.h)-y
            self.bbox.x = x
            self.bbox.y = y
            self.clear_canvas_n_draw()
            return
        if self.BR:
            self.bbox.w = x-self.bbox.x
            self.bbox.h = y-self.bbox.y
            self.clear_canvas_n_draw()
            return
        if self.TR:
            self.bbox.h = (self.bbox.y+self.bbox.h)-y
            self.bbox.y = y
            self.bbox.w = x-self.bbox.x
            self.clear_canvas_n_draw()
            return
        if self.BL:
            self.bbox.w = (self.bbox.x+self.bbox.w)-x
            self.bbox.x = x
            self.bbox.h = y-self.bbox.y
            self.clear_canvas_n_draw()
            return
        
        if self.TM:
            self.bbox.h = (self.bbox.y+self.bbox.h)-y
            self.bbox.y = y
            self.clear_canvas_n_draw()
            return
        if self.BM:
            self.bbox.h = y-self.bbox.y
            self.clear_canvas_n_draw()
            return
        
        if self.LM:
            self.bbox.w = (self.bbox.x+self.bbox.w)-x
            self.bbox.x = x
            self.clear_canvas_n_draw()
            return
        if self.RM:
            self.bbox.w = x-self.bbox.x
            self.clear_canvas_n_draw()
            return

    def mouse_double_click(self, x,y):
        if self.bbox.is_inside(x,y):
            self.is_return=True
            cv2.destroyWindow(self.window_name)

# def get_xyhw(face_pts:np):
#     min_x, min_y = min(face_pts[:, 0]), min(face_pts[:, 1])
#     max_x, max_y = max(face_pts[:, 0]), max(face_pts[:, 1])

#     # margin = 0.13
#     # min_x = int(min_x * (1-margin))
#     # min_y = int(min_y * (1-margin))
#     # max_x = int(max_x * (1+margin))
#     # max_y = int(max_y * (1+margin))
#     height = max_y-min_y
#     width = max_x-min_x
#     return int(min_x), int(min_y), int(width), int(height)

# def draw_bbox(img, bbox, pc=(0,0,255), radius=2, lc=(0,255,0), thickness=2):
#     print(bbox)
#     return img


# def closest_node(xy, pts):
#     #search the list of nodes for the one closest to node, return the name
#     dist_2 = np.sqrt(np.sum((pts - np.array(xy).reshape((-1, 2)))**2, axis=1))
#     if (dist_2[np.argmin(dist_2)] > 20):
#         return -1
#     return np.argmin(dist_2)

img0 = cv2.imread("sample/prev-0-w.png")
img = np.copy(img0)
pts = np.loadtxt("sample/prev-0_face_open_mouth.txt")
# bbox = get_xyhw(pts)

def click_adjust_wireframe(event, x, y, flags, param):
    global img, pts, node, bbox

    def update_img(node, button_up=False):
        global img, pts, bbox

        # update carton points object and get fresh pts list
        pts[node, 0], pts[node, 1] = x, y

        img = draw_bbox(img, pts)

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

h,w,c = img.shape
# rectI = selectinwindow.DragRectangle(img, 'app', w, h)
app_window = AppWindow(img, 'app', Rect(0,0,w,h), Rect(25,25,w-100,h-100))

cv2.namedWindow('app')
cv2.setMouseCallback('app', app_window.drag_rect, app_window)
app_window.clear_canvas_n_draw()

while True:
    key = cv2.waitKey(1) & 0xFF
cv2.destroyAllWindows()