import os
import numpy as np
import cv2

img0 = cv2.imread("sample/prev-0-w.png")
pts0 = np.loadtxt("sample/prev-0_face_open_mouth.txt")[:, :2]
pts = np.copy(pts0)
h,w,c = img0.shape

hold = False
mode = 'none'
ox, oy = 0, 0
dx, dy, dw, dh = 0, 0, 0, 0

def get_bbox(face_pts):
    min_x, min_y = face_pts[:,0].min(), face_pts[:,1].min()
    max_x, max_y = face_pts[:,0].max(), face_pts[:,1].max()

    h,w = max_y-min_y,max_x-min_x

    return np.array([min_x, min_y, w, h]).astype(int)

face_bbox0 = get_bbox(pts0)
face_bbox = np.copy(face_bbox0)

def mouse_callback(event, x,y, flags, params):
    global pts, pts0, face_bbox, face_bbox0
    global hold, mode, ox, oy, dx, dy, dw, dh

    if event == cv2.EVENT_LBUTTONDOWN:
        hold = True
        ox, oy = x, y
        set_mode(x,y)
        print(mode)
        return
    if hold and event == cv2.EVENT_MOUSEMOVE:
        if mode == 'drag':
            dx, dy = x-ox, y-oy
        if mode == 'tl':
            dx = x-ox
            dw = ox-x
            dy = y-oy
            dh = oy-y
        if mode == 'tm':
            dy = y-oy
            dh = oy-y
        if mode == 'tr':
            dw = x-ox
            dy = y-oy
            dh = oy-y
        if mode == 'lm':
            dx = x-ox
            dw = ox-x
        if mode == 'rm':
            dw = x-ox
        if mode == 'bl':
            dx = x-ox
            dw = ox-x
            dh = y-oy
        if mode == 'bm':
            dh = y-oy
        if mode == 'br':
            dw = x-ox
            dh = y-oy
        return
    if event == cv2.EVENT_LBUTTONUP:
        hold = False

        face_bbox[0] += dx
        face_bbox[1] += dy
        face_bbox[2] += dw
        face_bbox[3] += dh
        
        sw,sh = face_bbox[2]/face_bbox0[2], face_bbox[3]/face_bbox0[3]
        pts[:,0] = (pts0[:,0] - face_bbox0[0])*sw + face_bbox[0]
        pts[:,1] = (pts0[:,1] - face_bbox0[1])*sh + face_bbox[1]
        
        mode = 'none'
        ox, oy = 0, 0
        dx, dy, dw, dh = 0, 0, 0, 0
        return

def set_mode(px,py):
    global mode
    x,y,w,h = face_bbox
    for (m, cx,cy) in [
        ('tl',x,y), ('tm', x+w/2,y), ('tr', x+w,y),
        ('lm',x,y+h/2), ('rm', x+w,y+h/2),
        ('bl',x,y+h), ('bm',x+w/2,y+h), ('br',x+w,y+h),
    ]:
        if (cx-5<px<cx+5) and (cy-5<py<cy+5):
            mode = m
            return
    if (x < px < x+w) and (y < py < y+h):
        mode = 'drag'
        return

def draw_pts(img):
    global hold, mode, dx, dy, dw, dh
    global pts, face_bbox
    x,y,w,h = face_bbox
    ox,oy,x,y = x,y,x+dx,y+dy
    ow,oh,w,h = w,h,w+dw,h+dh
    sw, sh = w/ow, h/oh

    # draw landmarks
    for pt in pts:
        px, py = pt[0]-ox, pt[1]-oy
        px, py = px*sw+dx+ox,py*sh+dy+oy
        px, py = int(px), int(py)
        img = cv2.circle(img, (px,py), 2, (0,0,255), 2)

    # draw bbox
    img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    for (cx,cy) in [
        (x,y), (x+w/2,y), (x+w,y),
        (x,y+h/2), (x+w,y+h/2),
        (x,y+h), (x+w/2,y+h), (x+w,y+h),
    ]:
        cx,cy = int(cx),int(cy)
        img = cv2.rectangle(img, (cx-5,cy-5), (cx+5,cy+5), (0,255,0), 2)
    return img

def draw_image():
    img = np.copy(img0)
    img = draw_pts(img)
    cv2.imshow('app', img)


cv2.namedWindow('app')
cv2.setMouseCallback('app', mouse_callback, {})

while True:
    draw_image()
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
print(pts)