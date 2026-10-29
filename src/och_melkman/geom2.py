# Copyright (c) 2026 Institute of Mathematical and Computational Sciences (IMACS), HCMUT, VNU-HCM.
# Authors: Phong Thi Thu Huyen, Tran Anh Dung, Phan Thanh An. Released under the MIT License (see LICENSE).
import numpy as np,cv2
from . import geom
def _pad(mask):
    ys,xs=np.nonzero(mask); h,w=mask.shape; S=int(np.ceil(np.hypot(h,w)))+4
    big=np.zeros((S,S),np.float32); oy,ox=(S-h)//2,(S-w)//2; big[oy:oy+h,ox:ox+w]=mask
    return big,(xs.mean()+ox,ys.mean()+oy),S
def _crop(r):
    rows=np.nonzero(r.any(1))[0]; r=r[rows[0]:rows[-1]+1]; cols=np.nonzero(r.any(0))[0]; return r[:,cols[0]:cols[-1]+1]
def _sym(r):
    yy,xx=np.nonzero(r); c=xx.mean(); S=r.shape[1]
    mir=np.zeros_like(r); xm=np.round(2*c-xx).astype(int); ok=(xm>=0)&(xm<S); mir[yy[ok],xm[ok]]=True
    return (r&mir).sum()/max(1,(r|mir).sum())
def _nose_up(r):
    L=r.shape[0]; k=max(2,int(0.12*L))
    return r[::-1] if r[:k].sum(1).mean()>r[-k:].sum(1).mean() else r
def mrr_align(mask):
    """principal axis from the minimum-area rotated rectangle (MRR). returns {'mrr':..,'mrr_sym':..} aligned masks, MRR angle"""
    big,(cx,cy),S=_pad(mask)
    cnts,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    pts=np.vstack([c.reshape(-1,2) for c in cnts]).astype(np.float32)
    (rcx,rcy),(rw,rh),th=cv2.minAreaRect(pts)
    best=None
    for a in (th,-th,th+90,-th-90):
        r=_crop(geom._rot(big,a,(cx,cy),(S,S))>0.5); area=r.shape[0]*r.shape[1]
        if best is None or area<best[0]-1e-9: best=(area,a)
    a=best[1]; out=[]
    for aa in (a,a+90):
        r=_crop(geom._rot(big,aa,(cx,cy),(S,S))>0.5); out.append(r)
    tall=[r for r in out if r.shape[0]>=r.shape[1]]; tall=tall[0] if tall else out[0]
    symr=max(out,key=lambda r:_sym(np.pad(r,((0,0),(0,0)))))
    return _nose_up(tall).astype(np.uint8),_nose_up(symr).astype(np.uint8),a
