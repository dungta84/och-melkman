# Copyright (c) 2026 Institute of Mathematical and Computational Sciences (IMACS), HCMUT, VNU-HCM.
# Authors: Phong Thi Thu Huyen, Tran Anh Dung, Phan Thanh An. Released under the MIT License (see LICENSE).
import os,sys
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.join(ROOT,'src'))
DATA=os.environ.get('MTARSI_DIR',os.path.join(ROOT,'data','MTARSI-fixed')).rstrip('/\\')+'/'
RES=os.path.join(ROOT,'results')+'/'; WORK=os.path.join(ROOT,'work')+'/'; os.makedirs(WORK,exist_ok=True)
import pandas as pd,numpy as np,pickle,warnings; warnings.filterwarnings('ignore')
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold,GridSearchCV,StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix,accuracy_score
AL=sys.argv[1]; EXCL=os.environ.get('EXCL',''); REP=int(os.environ.get('REP',20)); SUBS=os.environ.get('SUBS','all,unique,qc').split(',')
df=pd.read_csv(RES+'features10.csv'); df=df[df.ok==1]
if EXCL: df=df[df.cls!=EXCL]
df=df.reset_index(drop=True)
classes=[c for c in ['B-1','B-52','C-130','C-135','B-2','C-17','C-5','P-3','KC-10','E-3'] if c in set(df.cls)]; K=len(classes); y=df.cls.map({c:i for i,c in enumerate(classes)}).values
fn=['LWRatio','ARatio','MaxDepth','XMDis','YMDis','SubDepth','XSDis','YSDis']
sets={'CH':[AL+'_CH_'+f for f in fn],'OCH':[AL+'_OCH_'+f for f in fn]}; 
subsets={'all':np.ones(len(df),bool),'unique':~df.md5.duplicated().values,'qc':((df.sym>=0.80)&(df[AL+'_valid']==1)&(df.detected==1)).values}
def run(mask,cols):
    X=df.loc[mask,cols].values; yy=y[mask]; g=df.loc[mask,'md5'].values
    P=[];R=[];F=[];ACC=[];CM=np.zeros((K,K))
    for seed in range(REP):
        pred=np.zeros(len(yy),int)
        for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(X,yy,g):
            gs=GridSearchCV(make_pipeline(StandardScaler(),SVC(kernel='rbf')),{'svc__C':[1,10,100],'svc__gamma':['scale',0.3,0.03]},cv=StratifiedKFold(3,shuffle=True,random_state=seed),n_jobs=1)
            gs.fit(X[tr],yy[tr]); pred[te]=gs.predict(X[te])
        p,r,f,s=precision_recall_fscore_support(yy,pred,labels=range(K),zero_division=0)
        P.append(p);R.append(r);F.append(f);ACC.append(accuracy_score(yy,pred)); CM+=confusion_matrix(yy,pred,labels=range(K))
    return dict(P=np.mean(P,0),R=np.mean(R,0),F=np.mean(F,0),acc=float(np.mean(ACC)),acc_sd=float(np.std(ACC)),CM=CM/REP,support=np.bincount(yy,minlength=K),n=int(mask.sum()),classes=classes)
out={}
for sn in SUBS:
    for fs,cols in sets.items():
        r=run(subsets[sn],cols); out[f'{sn}|{fs}']=r
        print(AL,EXCL or '7cls',sn,fs,'n=',r['n'],'acc=%.3f±%.3f'%(r['acc'],r['acc_sd']),'F1',np.round(r['F'],2),flush=True)
pickle.dump(out,open(RES+'eval10_%s%s_%s.pkl'%(AL,'_ex'+EXCL if EXCL else '','_'.join(SUBS)),'wb'))
