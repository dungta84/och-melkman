# Copyright (c) 2026 Institute of Mathematical and Computational Sciences (IMACS), HCMUT, VNU-HCM.
# Authors: Phong Thi Thu Huyen, Tran Anh Dung, Phan Thanh An. Released under the MIT License (see LICENSE).
import os,sys
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.join(ROOT,'src'))
DATA=os.environ.get('MTARSI_DIR',os.path.join(ROOT,'data','MTARSI-fixed')).rstrip('/\\')+'/'
RES=os.path.join(ROOT,'results')+'/'; WORK=os.path.join(ROOT,'work')+'/'; os.makedirs(WORK,exist_ok=True)
import pickle,json,cv2,numpy as np,pandas as pd,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R=DATA
import glob; d=pd.read_csv(RES+'features10.csv'); det={}
for _f in sorted(glob.glob(RES+'detections/det_*.json')): det.update(json.load(open(_f)))
d=d[d.ok==1]; import sys

ORDER=['B-1','B-52','C-130','C-135','C-17','C-5','P-3','KC-10','E-3','B-2']
KEY={'B-1':'B-1/B-1_081.jpg','B-52':'B-52/B-52_026.jpg','C-130':'C-130/C-130_038.jpg','C-135':'C-135/C-135_009.jpg','B-2':'B-2/B-2_069.jpg','C-17':'C-17/C-17_049.jpg','C-5':'C-5/C-5_100.jpg','P-3':'P-3/P-3_047.jpg','KC-10':'KC-10/KC-10_003.jpg','E-3':'E-3/E-3_002.jpg'}
FN=['LWRatio','ARatio','MaxDepth','XMDis','YMDis','SubDepth','XSDis','YSDis']
W,H=4.8,7.9
fig=plt.figure(figsize=(W,H)); rowh=1/5.0
for n,c in enumerate(ORDER):
    col,rw=divmod(n,5); k=KEY[c]; row=d[d.cls==c]
    im=cv2.cvtColor(cv2.imread(R+k),cv2.COLOR_BGR2RGB); b=det[k]['box'] or [0,0,im.shape[1],im.shape[0]]
    x0=col*0.5+0.005; y0=1-(rw+1)*rowh
    a1=fig.add_axes([x0,y0+0.055*1.0,0.22,rowh-0.085]); a2=fig.add_axes([x0+0.235,y0+0.055,0.25,rowh-0.085]); at=fig.add_axes([x0,y0,0.49,0.05]); at.axis('off')
    a1.imshow(im); a1.add_patch(plt.Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,ec='lime',lw=1.0)); a1.axis('off'); a1.set_title(k.replace('.jpg',''),fontsize=6,pad=2)
    r,P,poly,ch,g=pickle.load(open(WORK+'vis/'+k.replace('/','_')+'.pkl','rb'))
    a2.imshow(1-r,cmap='gray',alpha=0.5)
    q=np.array(P+[P[0]]); a2.plot(q[:,0]-.5,q[:,1]-.5,'g',lw=.4)
    q=np.array(poly+[poly[0]]); a2.plot(q[:,0]-.5,q[:,1]-.5,'b',lw=.8)
    q=np.array([tuple(p) for p in ch]+[tuple(ch[0])]); a2.plot(q[:,0]-.5,q[:,1]-.5,'r',lw=.8)
    rc=g['rect']; a2.plot([rc[0],rc[2],rc[2],rc[0],rc[0]],[rc[1],rc[1],rc[3],rc[3],rc[1]],'k',lw=.5); a2.axvline(g['xh'],color='k',ls=':',lw=.5)
    a2.plot(*g['A'],'ko',ms=2.5); a2.plot(*g['B'],'ko',ms=2.5); a2.plot(*g['p1'],'mo',ms=3); a2.plot(*g['p2'],'mo',ms=3)
    a2.set_aspect('equal'); a2.axis('off')
    v=[row[row.img==k]['S_OCH_'+f].iloc[0] for f in FN]
    at.text(0.5,0.5,'LW %.2f  AR %.2f  MaxD %.2f  XM %.2f\nYM %.2f  SubD %.2f  XS %.2f  YS %.2f'%tuple(v),fontsize=4.6,ha='center',va='center',linespacing=1.3)
fig.savefig(RES+'fig_aircraft_pipeline_onepage.png',dpi=330)
