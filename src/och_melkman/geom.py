# Copyright (c) 2026 Institute of Mathematical and Computational Sciences (IMACS), HCMUT, VNU-HCM.
# Authors: Phong Thi Thu Huyen, Tran Anh Dung, Phan Thanh An. Released under the MIT License (see LICENSE).
import numpy as np, cv2
from . import omelkman as om

# ---------- segmentation ----------
def _cc_center(fg,box):
    n,lab,st,cen=cv2.connectedComponentsWithStats(fg.astype(np.uint8),connectivity=4)
    if n<=1: return fg
    x1,y1,x2,y2=box; cx,cy=(x1+x2)/2,(y1+y2)/2
    # score = area, with bonus if component covers box centre region
    best,bs=1,-1
    for k in range(1,n):
        a=st[k,cv2.CC_STAT_AREA]
        reg=(lab==k)[int(cy-0.15*(y2-y1)):int(cy+0.15*(y2-y1))+1,int(cx-0.15*(x2-x1)):int(cx+0.15*(x2-x1))+1].mean() if a>20 else 0
        sc=a*(1+3*reg)
        if sc>bs: bs,best=sc,k
    return (lab==best).astype(np.uint8)
def _fill(fg):
    inv=(1-fg).astype(np.uint8); n,lab=cv2.connectedComponents(np.pad(inv,1,constant_values=1),connectivity=4); lab=lab[1:-1,1:-1]
    return (lab!=lab[0,0]).astype(np.uint8)
def segment(im, box, iters=6):
    """GrabCut seeded by the YOLO box: outside box = sure background, centre disc = sure foreground."""
    h,w=im.shape[:2]
    if box is None: box=[0,0,w,h]
    x1,y1,x2,y2=[int(round(v)) for v in box]; x2=max(x2,x1+4); y2=max(y2,y1+4)
    bw,bh=x2-x1,y2-y1; cx,cy=(x1+x2)//2,(y1+y2)//2
    m=np.full((h,w),cv2.GC_BGD,np.uint8); m[y1:y2,x1:x2]=cv2.GC_PR_BGD
    yy,xx=np.mgrid[0:h,0:w]
    r=np.sqrt(((xx-cx)/(bw/2+1e-6))**2+((yy-cy)/(bh/2+1e-6))**2)
    m[(r<0.55)&(m!=cv2.GC_BGD)]=cv2.GC_PR_FGD; m[r<0.12]=cv2.GC_FGD
    bgd=np.zeros((1,65));fgd=np.zeros((1,65))
    cv2.grabCut(im,m,None,bgd,fgd,iters,cv2.GC_INIT_WITH_MASK)
    g=((m==1)|(m==3)).astype(np.uint8)
    g=cv2.morphologyEx(g,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    g=_fill(_cc_center(g,[x1,y1,x2,y2]))
    return g,'grabcut'

# ---------- orientation ----------
def _rot(mask,ang,center,size):
    M=cv2.getRotationMatrix2D(center,ang,1.0)
    return cv2.warpAffine(mask.astype(np.float32),M,size,flags=cv2.INTER_LINEAR)
def align(mask):
    ys,xs=np.nonzero(mask); 
    h,w=mask.shape; S=int(np.ceil(np.hypot(h,w)))+4
    big=np.zeros((S,S),np.float32); oy,ox=(S-h)//2,(S-w)//2; big[oy:oy+h,ox:ox+w]=mask
    cy,cx=ys.mean()+oy,xs.mean()+ox
    def score(a):
        r=_rot(big,a,(cx,cy),(S,S))>0.5
        yy,xx=np.nonzero(r)
        if len(xx)==0: return 0
        # mirror about vertical line through centroid column
        c=xx.mean(); mir=np.zeros_like(r)
        xm=np.round(2*c-xx).astype(int); ok=(xm>=0)&(xm<S)
        mir[yy[ok],xm[ok]]=True
        return (r&mir).sum()/max(1,(r|mir).sum())
    angs=np.arange(0,180,3.0); sc=[score(a) for a in angs]; a0=angs[int(np.argmax(sc))]
    fine=np.arange(a0-3,a0+3.01,0.5); sf=[score(a) for a in fine]; a1=fine[int(np.argmax(sf))]
    r=_rot(big,a1,(cx,cy),(S,S))>0.5
    # head direction: narrower end = nose
    rows=np.nonzero(r.any(1))[0]; r=r[rows[0]:rows[-1]+1]
    cols=np.nonzero(r.any(0))[0]; r=r[:,cols[0]:cols[-1]+1]
    L=r.shape[0]; k=max(2,int(0.12*L))
    wtop=r[:k].sum(1).mean(); wbot=r[-k:].sum(1).mean()
    if wtop>wbot: r=r[::-1]   # make nose (narrow end) the top
    return r.astype(np.uint8), a1, max(sc+sf)

def tidy(r):
    """remove diagonal pinches, keep largest 4-connected comp, fill holes -> simple rectilinear region"""
    r=r.copy().astype(np.uint8)
    for _ in range(2000):
        a=r[:-1,:-1];b=r[:-1,1:];c=r[1:,:-1];d=r[1:,1:]
        p1=(a==1)&(d==1)&(b==0)&(c==0); p2=(b==1)&(c==1)&(a==0)&(d==0)
        if not (p1.any() or p2.any()): break
        ys,xs=np.nonzero(p1|p2)
        for y,x in zip(ys,xs): r[y,x+1]=1;r[y+1,x]=1;r[y,x]=1;r[y+1,x+1]=1
    n,lab,st,_=cv2.connectedComponentsWithStats(r,connectivity=4)
    if n>2:
        k=1+np.argmax(st[1:,cv2.CC_STAT_AREA]); r=(lab==k).astype(np.uint8)
    inv=(1-r).astype(np.uint8); n,lab=cv2.connectedComponents(np.pad(inv,1,constant_values=1),connectivity=4); lab=lab[1:-1,1:-1]
    outer=lab[0,0]
    r=(lab!=outer).astype(np.uint8)
    return r

# ---------- rectilinear boundary ----------
def outer_polygon(r):
    """vertices (x,y) in pixel-corner coords of the outer boundary of region r (CCW in y-up image sense), collinear removed"""
    r=np.pad(r,1).astype(np.uint8); H,W=r.shape
    nxt={}
    ys,xs=np.nonzero(r)
    for y,x in zip(ys,xs):
        if r[y-1,x]==0: nxt[(x,y)]=(x+1,y)          # top edge, going right
        if r[y,x+1]==0: nxt[(x+1,y)]=(x+1,y+1)      # right edge, going down
        if r[y+1,x]==0: nxt[(x+1,y+1)]=(x,y+1)      # bottom edge, going left
        if r[y,x-1]==0: nxt[(x,y+1)]=(x,y)          # left edge, going up
    start=min(nxt); loop=[start]; cur=nxt[start]
    while cur!=start:
        loop.append(cur); cur=nxt[cur]
        if len(loop)>10**6: raise RuntimeError('loop')
    if len(loop)!=len(nxt): raise RuntimeError('boundary has several loops/pinch')
    # remove collinear
    out=[]; n=len(loop)
    for i in range(n):
        p0,p1,p2=loop[i-1],loop[i],loop[(i+1)%n]
        if (p1[0]-p0[0])*(p2[1]-p1[1])-(p1[1]-p0[1])*(p2[0]-p1[0])!=0: out.append((p1[0]-1,p1[1]-1))
    return out

def shoelace(pts):
    a=0
    for i in range(len(pts)):
        x1,y1=pts[i];x2,y2=pts[(i+1)%len(pts)]; a+=x1*y2-x2*y1
    return a/2

# ---------- independent COCH by closure ----------
def coch_closure(r):
    r=r.astype(bool).copy()
    while True:
        old=r.copy()
        for ax in (0,1):
            a=r if ax==1 else r.T
            for i in range(a.shape[0]):
                idx=np.nonzero(a[i])[0]
                if len(idx): a[i,idx[0]:idx[-1]+1]=True
        if (r==old).all(): return r

# ---------- features ----------
def dedupe(pts):
    o=[]
    for p in pts:
        if not o or o[-1]!=p: o.append(p)
    if len(o)>1 and o[0]==o[-1]: o.pop()
    return o
def feats_from_hull(P,hullpoly,hull_is_coch,Hs):
    """P: list of (x,y) polygon vertices (image coords, y down, head at top); hullpoly: list of (x,y)"""
    P=np.array(P,float); xmin,ymin=P.min(0); xmax,ymax=P.max(0)
    Wd,Ln=xmax-xmin,ymax-ymin
    top=P[P[:,1]==ymin]; xh=top[:,0].mean()
    S=Wd*Ln
    area_h=abs(shoelace([tuple(p) for p in hullpoly]))
    A=np.array([xmin,ymin]);B=np.array([xmin,ymax])
    hp=np.array(hullpoly,float)
    left=hp[hp[:,0]<=xh]
    cand=left
    if hull_is_coch:
        # reflex vertices of the hull polygon
        n=len(hp); sgn=np.sign(shoelace([tuple(p) for p in hp])); refl=[]
        for i in range(n):
            p0,p1,p2=hp[i-1],hp[i],hp[(i+1)%n]
            cr=(p1[0]-p0[0])*(p2[1]-p1[1])-(p1[1]-p0[1])*(p2[0]-p1[0])
            if cr*sgn<0: refl.append(p1)
        refl=np.array(refl) if refl else np.zeros((0,2)); 
        refl=refl[refl[:,0]<=xh] if len(refl) else refl
        if len(refl)>0: cand=refl
    p1=cand[np.argmin(np.linalg.norm(cand-A,axis=1))]
    p2=cand[np.argmin(np.linalg.norm(cand-B,axis=1))]
    d=lambda u,v: float(np.linalg.norm(np.array(u)-np.array(v)))
    f=dict(LWRatio=Ln/Wd, ARatio=area_h/S,
           MaxDepth=d(A,p1)/Ln, XMDis=abs(p1[0]-xh)/Ln, YMDis=abs(p1[1]-ymin)/Ln,
           SubDepth=d(B,p2)/Ln, XSDis=abs(p2[0]-xh)/Ln, YSDis=abs(p2[1]-ymin)/Ln)
    return f,dict(A=A.tolist(),B=B.tolist(),p1=p1.tolist(),p2=p2.tolist(),xh=float(xh),rect=[xmin,ymin,xmax,ymax])

def coch_polygon(P):
    """run the author's O-Melkman on polygon vertex list; return (extreme_vertices, hull_polygon_vertices, status)"""
    pts=[om.Point(x,y) for x,y in P]
    om.operation_counts.update(check_vertex=0,add_remove_vertex=0)
    d=om.ortho_hull(pts)
    ext=[(p.x,p.y) for p in d]
    poly=dedupe([(p.x,p.y) for p in om.adding_points(list(d))])
    return ext,poly,om.is_ortho_convex(d)
