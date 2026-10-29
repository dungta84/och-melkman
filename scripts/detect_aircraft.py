# Copyright (c) 2026 Institute of Mathematical and Computational Sciences (IMACS), HCMUT, VNU-HCM.
# Authors: Phong Thi Thu Huyen, Tran Anh Dung, Phan Thanh An. Released under the MIT License (see LICENSE).
import os,sys
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.join(ROOT,'src'))
DATA=os.environ.get('MTARSI_DIR',os.path.join(ROOT,'data','MTARSI-fixed')).rstrip('/\\')+'/'
RES=os.path.join(ROOT,'results')+'/'; WORK=os.path.join(ROOT,'work')+'/'; os.makedirs(WORK,exist_ok=True)
from ultralytics import YOLO
import glob,json,numpy as np,cv2
R=DATA
m=YOLO(os.environ.get('YOLO_WEIGHTS',os.path.join(ROOT,'weights','best.pt')))
C=sys.argv[1]
res={}
if os.path.exists(RES+'detections/det_'+C+'.json'): res=json.load(open(RES+'detections/det_'+C+'.json'))
C=sys.argv[1]
for c in [C]:
    for f in sorted(glob.glob(R+c+'/*.jpg')):
        key=c+'/'+os.path.basename(f)
        if key in res: continue
        im=cv2.imread(f); h,w=im.shape[:2]
        edge=np.concatenate([im[0],im[-1],im[:,0],im[:,-1]]); bg=np.median(edge,0).astype(int).tolist()
        best=None
        for k in (3,5):
            py,px=h*(k-1)//2,w*(k-1)//2
            img=cv2.copyMakeBorder(im,py,py,px,px,cv2.BORDER_CONSTANT,value=bg)
            r=m.predict(img,imgsz=960,conf=0.1,verbose=False,device='cpu')[0]
            for cf,b in zip(r.boxes.conf.tolist(),r.boxes.xyxy.tolist()):
                x1,y1,x2,y2=b[0]-px,b[1]-py,b[2]-px,b[3]-py
                cx,cy=(x1+x2)/2,(y1+y2)/2
                if 0<=cx<=w and 0<=cy<=h and (best is None or cf>best['conf']):
                    best=dict(conf=cf,box=[max(0,x1),max(0,y1),min(w,x2),min(h,y2)],k=k)
            if best: break
        res[key]=best or dict(conf=0,box=None,k=None)
        res[key]['size']=[w,h]
        json.dump(res,open(RES+'detections/det_'+C+'.json','w'))
        print(key,res[key],flush=True)
print("DONE")
