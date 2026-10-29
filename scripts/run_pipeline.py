# Copyright (c) 2026 Institute of Mathematical and Computational Sciences (IMACS), HCMUT, VNU-HCM.
# Authors: Phong Thi Thu Huyen, Tran Anh Dung, Phan Thanh An. Released under the MIT License (see LICENSE).
import os,sys
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.join(ROOT,'src'))
DATA=os.environ.get('MTARSI_DIR',os.path.join(ROOT,'data','MTARSI-fixed')).rstrip('/\\')+'/'
RES=os.path.join(ROOT,'results')+'/'; WORK=os.path.join(ROOT,'work')+'/'; os.makedirs(WORK,exist_ok=True)
import json,cv2,numpy as np,hashlib,pickle,traceback
from och_melkman import geom,segsel,geom2
R=DATA
C=sys.argv[1]
det={}
import glob
for f in sorted(glob.glob(RES+'detections/det_*.json')): det.update(json.load(open(f)))
outf=WORK+'p_%s.jsonl'%C
res={json.loads(l)['img']:1 for l in open(outf)} if os.path.exists(outf) else {}
os.makedirs(WORK+'vis',exist_ok=True)
def feats(r,pref,rec):
    r=geom.tidy(r)
    try: P=geom.outer_polygon(r)
    except RuntimeError:
        r=geom.tidy(cv2.morphologyEx(r.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))); P=geom.outer_polygon(r)
    ext,poly,isoc=geom.coch_polygon(P)
    clo=int(geom.coch_closure(r).sum()); valid=int(abs(abs(geom.shoelace(poly))-clo)<1e-9 and isoc)
    hull=cv2.convexHull(np.array(P,np.float32)).reshape(-1,2); ch=[tuple(p) for p in hull]
    f1,_=geom.feats_from_hull(P,ch,False,None); f2,g=geom.feats_from_hull(P,poly,True,None)
    for n,v in f1.items(): rec[pref+'CH_'+n]=v
    for n,v in f2.items(): rec[pref+'OCH_'+n]=v
    rec[pref+'valid']=valid; rec[pref+'nv']=len(P); rec[pref+'k']=len(ext)
    return r,P,poly,ch,g
for k in sorted(det):
    if not k.startswith(C+'/') or k in res: continue
    im=cv2.imread(R+k); box=det[k]['box']
    rec=dict(img=k,cls=C,detected=int(box is not None),md5=hashlib.md5(open(R+k,'rb').read()).hexdigest()); vis=None
    try:
        (sym,meth,fg,r,ang),tab=segsel.select(im,box)
        rec['seg_method']=meth; rec['sym']=round(float(sym),3)
        m=segsel.normalise(fg).astype(np.uint8)
        vis=feats(r,'S_',rec)
        rm,rms,a=geom2.mrr_align(m); rec['mrr_angle']=round(float(a),1)
        feats(rm,'M_',rec); feats(rms,'MS_',rec); rec['ok']=1
    except Exception as e:
        rec['ok']=0; rec['err']=repr(e)[:100]
    res[k]=1
    with open(outf,'a') as fo: fo.write(json.dumps(rec)+'\n')
    if vis is not None: pickle.dump(vis,open(WORK+'vis/'+k.replace('/','_')+'.pkl','wb'))
print(C,len(res))
