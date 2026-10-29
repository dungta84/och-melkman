# Copyright (c) 2026 Institute of Mathematical and Computational Sciences (IMACS), HCMUT, VNU-HCM.
# Authors: Phong Thi Thu Huyen, Tran Anh Dung, Phan Thanh An. Released under the MIT License (see LICENSE).
import cv2,numpy as np
from . import geom
def _post(g,box):
    g=cv2.morphologyEx(g.astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    if g.sum()<30: return None
    g=geom._fill(geom._cc_center(g,box)); return g if g.sum()>=30 else None
def candidates(im,box):
    h,w=im.shape[:2]
    if box is None: box=[0,0,w,h]
    x1,y1,x2,y2=[int(round(v)) for v in box]; x2=max(x2,x1+4); y2=max(y2,y1+4)
    c={}
    g,_=geom.segment(im,box,6); c['gc_seed']=g
    g,_=geom.segment(im,box,2); c['gc_seed2']=g
    # gray Otsu, both polarities, inside box
    gray=cv2.GaussianBlur(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),(3,3),0)
    t,_=cv2.threshold(gray[y1:y2,x1:x2],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    win=np.zeros((h,w),bool); win[y1:y2,x1:x2]=True
    c['otsu_dark']=((gray<t)&win); c['otsu_light']=((gray>=t)&win)
    # k-means in Lab (k=2,3), centre cluster
    lab=cv2.cvtColor(cv2.GaussianBlur(im,(3,3),0),cv2.COLOR_BGR2LAB).reshape(-1,3).astype(np.float32)
    crit=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.5)
    cy,cx=(y1+y2)//2,(x1+x2)//2; ry,rx=int(0.12*(y2-y1))+1,int(0.12*(x2-x1))+1
    # GrabCut initialised with the YOLO rectangle only
    try:
        m=np.zeros((h,w),np.uint8); bgd=np.zeros((1,65));fgd=np.zeros((1,65))
        cv2.grabCut(im,m,(x1,y1,x2-x1,y2-y1),bgd,fgd,5,cv2.GC_INIT_WITH_RECT); c['gc_rect']=((m==1)|(m==3))&win
    except cv2.error: pass
    for k in (2,3,4):
        _,lb,_=cv2.kmeans(lab,k,None,crit,2,cv2.KMEANS_PP_CENTERS); lb=lb.reshape(h,w)
        cnt=np.bincount(lb[max(0,cy-ry):cy+ry,max(0,cx-rx):cx+rx].ravel(),minlength=k)
        c[f'km{k}']=((lb==np.argmax(cnt))&win)
    out={}
    for n,g in c.items():
        p=_post(g,[x1,y1,x2,y2])
        if p is not None: out[n]=p
    return out,[x1,y1,x2,y2]
def normalise(fg):
    ys,xs=np.nonzero(fg); crop=fg[ys.min():ys.max()+1,xs.min():xs.max()+1].astype(np.float32)
    sc=240/max(crop.shape)
    return cv2.resize(crop,(max(2,int(round(crop.shape[1]*sc))),max(2,int(round(crop.shape[0]*sc)))),interpolation=cv2.INTER_LINEAR)>0.5
def select(im,box):
    cands,b=candidates(im,box); area_box=max(1,(b[2]-b[0])*(b[3]-b[1])); best=None; table={}
    for n,g in cands.items():
        frac=g.sum()/area_box
        cnts,_=cv2.findContours(g.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); hullA=cv2.contourArea(cv2.convexHull(max(cnts,key=cv2.contourArea))); sol=g.sum()/max(1,hullA)
        if not (0.10<=frac<=0.62) or sol>0.80: table[n]=(None,round(frac,2),round(sol,2)); continue
        r,ang,sym=geom.align(normalise(g).astype(np.uint8)); table[n]=(round(float(sym),3),round(frac,2))
        if best is None or sym>best[0]: best=(sym,n,g,r,ang)
    if best is None:   # fall back to seeded GrabCut regardless of plausibility
        g=cands.get('gc_seed'); 
        if g is None: raise RuntimeError('no candidate')
        r,ang,sym=geom.align(normalise(g).astype(np.uint8)); best=(sym,'gc_seed(fallback)',g,r,ang)
    return best,table
