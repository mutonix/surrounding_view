import cv2
import numpy as np

def pts_angle_ordered(points, car_center=(300, 285), var_thres=0.01):
    
    car_center = np.array(car_center)
    vectors = points - car_center
    
    cplx = np.array([np.complex(*vec) for vec in vectors])
    angles = np.angle(cplx)
    points = points[np.argsort(angles)]
    
    ymax_order_mask = np.argsort(points[:, 1])
    p_ordered = points[ymax_order_mask] 
    slope_ang = []
    for i, _ in enumerate(p_ordered[:-1]):
        ang = np.arccos((p_ordered[i+1, 0] - p_ordered[i, 0]) / cal_dist(p_ordered[i+1, :], p_ordered[i, :]))
        slope_ang.append(ang)
    slope_var = np.var(slope_ang)

    if slope_var < var_thres:
        points = p_ordered
        
    return points

def cal_dist(point1, point2):
    return np.sqrt(((point1 - point2) ** 2).sum(axis=-1))

def find_neighbour_point(points, car_center=(300, 285), var_thres=0.01):
    
    if len(points) >= 2:    
        p_ordered = pts_angle_ordered(points, car_center, var_thres)
        pts_pair = []
        pdist = []
        for i, _ in enumerate(p_ordered[:-1]):
            dist = cal_dist(p_ordered[i][:2], p_ordered[i+1][:2])
            pts_pair.append((p_ordered[i][:2], p_ordered[i+1][:2]))
            pdist.append(dist)
        
        pts_pair = np.array(pts_pair)
        pdist = np.array(pdist)
    else: 
        pts_pair = None
        pdist = None        

    return pts_pair, pdist

def ps_retrieval(pts_pair, pdist, verPS_edge=[100 ,200], parPS_edge=[250, 400], ifpar=True):

    verPS_edge = np.array(verPS_edge)
    parPS_edge = np.array(parPS_edge) if ifpar else np.array([1000, 0])
    
    assert parPS_edge[0] >= verPS_edge[1], '输入正确停车位宽度或阈值'
    
    if pdist is not None:
        ps_detected = np.append(pts_pair, np.zeros((len(pdist), 1, 2)), axis=1)
    
        verPS_mask = np.logical_and(pdist >= verPS_edge[0], pdist <= verPS_edge[1])
        parPS_mask = np.logical_and(pdist >= parPS_edge[0], pdist <= parPS_edge[1]) 
        
        ver_PS = ps_detected[verPS_mask]
        ver_PS[:, 2, 0] = 1 # 垂直车位
        ps_detected[verPS_mask] = ver_PS
        par_PS = ps_detected[parPS_mask]
        par_PS[:, 2, 0] = 2 # 平行车位 
        ps_detected[parPS_mask] = par_PS # 平行车位 
    
        ps_num = np.zeros(3, dtype=np.int)
        ps_num[1] = len(ps_detected[np.where(ps_detected[:, 2, 0]==1)])
        ps_num[2] = len(ps_detected[np.where(ps_detected[:, 2, 0]==2)])
        ps_num[0] = ps_num[1:].sum()
        
        ps_detected = ps_detected[ps_detected[:, 2, :].sum(-1) != 0]

        if not np.any(ps_detected):
            ps_detected = None
    else:
        ps_num, ps_detected = None, None
    
    return ps_num, ps_detected

def get_ps_poly(ps_detected, verPS_depth=220, parPS_depth=120):
    
    ps_num = ps_detected.shape[0]
    ver_mask = ps_detected[:, 2, 0] == 1
    par_mask = ps_detected[:, 2, 0] == 2

    ver_ps = ps_detected[ver_mask] 
    par_ps = ps_detected[par_mask] 

    ver_Lps, ver_Rps = poly_xy(ver_ps, verPS_depth) if np.any(ver_mask) else (np.empty((1, 5, 2)), np.empty((1, 5, 2)))
    par_Lps, par_Rps = poly_xy(par_ps, parPS_depth) if np.any(par_mask) else (None, None)

    L_ps = np.append(ver_Lps, par_Lps, axis=0) if par_Lps is not None else ver_Lps
    R_ps = np.append(ver_Rps, par_Rps, axis=0) if par_Rps is not None else ver_Rps

    L_ps = np.delete(L_ps, 0, axis=0) if not np.any(ver_mask) else L_ps
    R_ps = np.delete(R_ps, 0, axis=0) if not np.any(ver_mask) else R_ps

    return L_ps, R_ps  

def poly_xy(ps_detected, depth):   
    
    ymax_order_arg = np.argsort(ps_detected[:, :2, 1])
    for i, pts in enumerate(ps_detected):
        temp = pts[:2]
        temp = temp[ymax_order_arg[i]]
        ps_detected[i, :2] = temp

    ps_detected = np.insert(ps_detected, 2, np.zeros((2, len(ps_detected), 2)), axis=1)
    
    angle = np.arccos((ps_detected[:, 1, 0] - ps_detected[:, 0, 0]) / cal_dist(ps_detected[:, 0], ps_detected[:, 1])).reshape(-1, 1)
    x_shift = depth * np.sin(angle)
    y_shift = depth * np.cos(angle)
    
    L_ps ,R_ps = ps_detected.copy(), ps_detected.copy()

    L_ps[:, [2, 3], 0] = L_ps[:, [0, 1], 0] - x_shift
    L_ps[:, [2, 3], 1] = L_ps[:, [0, 1], 1] + y_shift
        
    R_ps[:, [2, 3], 0] = R_ps[:, [0, 1], 0] + x_shift
    R_ps[:, [2, 3], 1] = R_ps[:, [0, 1], 1] - y_shift
        
    return L_ps, R_ps

def get_center(pkslots):
    
    center = (pkslots[:, 0, :] + pkslots[:, 3, :]) / 2
    pkslots = np.insert(pkslots, 4, center, axis=1)
    
    return center, pkslots

def center_LR2F(L_center, R_center, car_center=(300, 285)):
    
    L2car = cal_dist(L_center, car_center)
    R2car = cal_dist(R_center, car_center)
    
    mask1 = L2car > R2car
    F_center = R_center.copy()
    F_center[mask1] = L_center[mask1]
    
    return F_center

def get_finalps(F_center, L_psc, R_psc):
    
    F_mask = np.all(F_center == L_psc[:, 4], axis=-1)
    F_psc = R_psc.copy()
    F_psc[F_mask] = L_psc[F_mask]
    
    return F_psc

def plot_ps(img, final_ps, colors=None):
    
    img_h, img_w = img.shape[:2]
    np.random.seed(626)
    colors = colors or [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(final_ps))]
    
    ps_rect = np.round(final_ps).astype(np.int32)
    ps_rect = np.ascontiguousarray(ps_rect[:, [0, 2, 3, 1]]).reshape(-1, 1,4,2)
    
    for i, pts in enumerate(ps_rect):
        if final_ps[i, 5, 1] == 0:
            cv2.polylines(img, pts, True, colors[i], 5)
        else:
            cv2.polylines(img, pts, True, colors[i], 5)
            cv2.polylines(img, np.ascontiguousarray(pts[:, [0, 2, 3, 1]]), True, colors[i], 5)

    return img

def get_box(result, box_len=10):

    box = np.empty((len(result), 4))

    box[:, 0] = result[:, 0] - box_len
    box[:, 1] = result[:, 1] - box_len
    box[:, 2] = result[:, 0] + box_len
    box[:, 3] = result[:, 1] + box_len
    
    return box

def get_ps_coord(result, img, car_center=(290, 400)):

    h = img.shape[1]
    result_coord = np.zeros_like(result)
    result_coord[:, 0] = (result[:, 0] - car_center[0]) / 100
    result_coord[:, 1] = (h - result[:, 1] - car_center[1]) / 100

    return result_coord

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [np.random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 6, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, (c1[0] - 20 , c1[1]) , (c2[0] - 20, c2[1]), color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0] - 20 , c1[1] - 2), 0, tl / 6, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def car_occupy_detect(car_mask, final_ps, car_center=(290, 400), occupy_thres=0.5, thres_decay=True, dist_thres=2000):

    ps_rect = np.round(final_ps).astype(np.int32)
    ps_rect = np.ascontiguousarray(ps_rect[:, [0, 2, 3, 1]]).reshape(-1, 1,4,2)

    dist = cal_dist(car_center, final_ps[:, 4])
    ratio = 1 / (np.exp(dist / dist_thres)) if thres_decay else np.ones_like(dist)
    occupy_thres = occupy_thres * ratio

    for i, rect in enumerate(ps_rect):
        blank = np.zeros_like(car_mask)
        cv2.fillPoly(blank, rect, (255, 255, 255))
        ps_mask = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)
        and_mask = cv2.bitwise_and(car_mask, ps_mask)
        thres = and_mask.sum() / ps_mask.sum()

        if thres > occupy_thres[i]:
            final_ps[i, 5, 1] = 1

    return final_ps

def find_nearest(person_mask, car_center=(290, 400)):
        
    contours, _ = cv2.findContours(person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    nearest = []
    for contour in contours:
        contour = np.squeeze(contour, axis=-2)
        dist = ((contour - car_center) ** 2).sum(-1)
        n_idx = np.argmin(dist, axis=-1)
        nearest.append(contour[n_idx])

    return nearest

