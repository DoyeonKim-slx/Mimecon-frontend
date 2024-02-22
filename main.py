import os
import numpy as np
import cv2

img0 = cv2.imread("sample/prev-0-w.png")
pts0 = np.loadtxt("sample/prev-0_face_open_mouth.txt")[:, :2]
LANDMARK_INFO = {
    'eyebrow_l':[17,22],
    'eyebrow_r':[22,27],
    'nose':     [27,36],
    'eye_left': [36,42],
    'eye_right':[42,48],
    'mouth_in': [60,68], # 다른 부위 가리기 방지
    'mouth_out':[48,60],
    'jaw':      [ 0,17], # 다른 부위 가리기 방지
}

def get_bbox(face_pts):
    min_x, min_y = face_pts[:,0].min(), face_pts[:,1].min()
    max_x, max_y = face_pts[:,0].max(), face_pts[:,1].max()

    h,w = max_y-min_y,max_x-min_x

    return np.array([min_x, min_y, w, h]).astype(int)

class LandmarkGroup:
    mode: str = 'inactive'
    dx: int = 0
    dy: int = 0
    dw: int = 0
    dh: int = 0

    def __init__(self, landmarks):
        self.landmarks = landmarks.copy()
        self.bbox = get_bbox(landmarks)
        self.orig_bbox = self.bbox.copy()
        self.landmarks[:,0] -= self.orig_bbox[0]
        self.landmarks[:,1] -= self.orig_bbox[1]
    
    @property
    def x(self):
        return self.bbox[0]
    @x.setter
    def x(self, v):
        self.bbox[0] = v
    @property
    def y(self):
        return self.bbox[1]
    @x.setter
    def y(self, v):
        self.bbox[1] = v
    @property
    def w(self):
        return self.bbox[2]
    @x.setter
    def w(self, v):
        self.bbox[2] = v
    @property
    def h(self):
        return self.bbox[3]
    @x.setter
    def h(self, v):
        self.bbox[3] = v
    
    def get_pts(self):
        sw, sh = (self.bbox[2]+self.dw)/self.orig_bbox[2], (self.bbox[3]+self.dh)/self.orig_bbox[3]
        landmarks = self.landmarks.copy()
        landmarks[:, 0] = landmarks[:, 0]*sw + self.bbox[0] + self.dx
        landmarks[:, 1] = landmarks[:, 1]*sh + self.bbox[1] + self.dy
        return landmarks
    
    def check_active(self, px,py):
        x,y,w,h = self.bbox
        for (m, cx,cy) in [
            ('tl',x,y), ('tm', x+w/2,y), ('tr', x+w,y),
            ('lm',x,y+h/2), ('rm', x+w,y+h/2),
            ('bl',x,y+h), ('bm',x+w/2,y+h), ('br',x+w,y+h),
        ]:
            if (cx-5<px<cx+5) and (cy-5<py<cy+5):
                self.mode = m
                return True
        if (x < px < x+w) and (y < py < y+h):
            self.mode = 'drag'
            return True
        return False

    def set_delta(self, dx, dy):
        if self.mode == 'drag':
            self.dx, self.dy = dx, dy
        if self.mode == 'tl':
            self.dx = dx
            self.dw = -dx
            self.dy = dy
            self.dh = -dy
        if self.mode == 'tm':
            self.dy = dy
            self.dh = -dy
        if self.mode == 'tr':
            self.dw = dx
            self.dy = dy
            self.dh = -dy
        if self.mode == 'lm':
            self.dx = dx
            self.dw = -dx
        if self.mode == 'rm':
            self.dw = dx
        if self.mode == 'bl':
            self.dx = dx
            self.dw = -dx
            self.dh = dy
        if self.mode == 'bm':
            self.dh = dy
        if self.mode == 'br':
            self.dw = dx
            self.dh = dy

    def release(self):
        self.mode = 'inactive'
        self.bbox[0] += self.dx
        self.bbox[1] += self.dy
        self.bbox[2] += self.dw
        self.bbox[3] += self.dh
        self.dx = 0
        self.dy = 0
        self.dw = 0
        self.dh = 0
    
    def draw(self, img):
        x,y,w,h = self.bbox[0],self.bbox[1],self.bbox[2],self.bbox[3]
        x,y = x+self.dx,y+self.dy
        w,h = w+self.dw,h+self.dh
        # draw bbox
        img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 1)
        for (cx,cy) in [
            (x,y), (x+w/2,y), (x+w,y),
            (x,y+h/2), (x+w,y+h/2),
            (x,y+h), (x+w/2,y+h), (x+w,y+h),
        ]:
            cx,cy = int(cx),int(cy)
            img = cv2.rectangle(img, (cx-5,cy-5), (cx+5,cy+5), (0,255,0), 2)
        
        # draw landmarks
        for pt in self.get_pts():
            px, py = pt[0], pt[1]
            px, py = int(px), int(py)
            img = cv2.circle(img, (px,py), 2, (0,0,255), 2)
        
        return img

landmark_groups = {}
for name, (start_i, end_i) in LANDMARK_INFO.items():
    landmark_groups[name] = LandmarkGroup(pts0[start_i:end_i])

hold = False
current = 'none'
ox, oy = 0, 0

def check_active(px,py):
    global landmark_groups, hold, current, ox, oy
    for name, group in landmark_groups.items():
        if group.check_active(px, py):
            hold = True
            current = name
            ox, oy = px, py
            return

def mouse_callback(event, px,py, flags, params):
    global landmark_groups
    global hold, current, ox, oy

    if event == cv2.EVENT_LBUTTONDOWN:
        check_active(px,py)
        return
    if hold and event == cv2.EVENT_MOUSEMOVE:
        dx, dy = px-ox, py-oy
        landmark_groups[current].set_delta(dx,dy)
        return
    if hold and event == cv2.EVENT_LBUTTONUP:
        landmark_groups[current].release()
        hold = False
        current = 'none'
        ox, oy = 0, 0
        return

def draw_image():
    global landmark_groups
    img = np.copy(img0)
    for groups in landmark_groups.values():
        img = groups.draw(img)
    cv2.imshow('app', img)

cv2.namedWindow('app')
cv2.setMouseCallback('app', mouse_callback, {})

while True:
    draw_image()
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()

new_pts = np.zeros_like(pts0)
for name, groups in landmark_groups.items():
    (start_i, end_i) = LANDMARK_INFO[name]
    new_pts[start_i:end_i] = groups.get_pts()
print(new_pts)