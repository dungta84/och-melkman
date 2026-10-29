# Copyright (c) 2026 Institute of Mathematical and Computational Sciences (IMACS), HCMUT, VNU-HCM.
# Authors: Phong Thi Thu Huyen, Tran Anh Dung, Phan Thanh An. Released under the MIT License (see LICENSE).
import os,sys
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.join(ROOT,'src'))
DATA=os.environ.get('MTARSI_DIR',os.path.join(ROOT,'data','MTARSI-fixed')).rstrip('/\\')+'/'
RES=os.path.join(ROOT,'results')+'/'; WORK=os.path.join(ROOT,'work')+'/'; os.makedirs(WORK,exist_ok=True)
import pickle,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
O=RES
ev=pickle.load(open(O+'eval10_S_all.pkl','rb')); ev.update(pickle.load(open(O+'eval10_S_unique_qc.pkl','rb')))
cl=['B-1','B-52','C-130','C-135','B-2','C-17','C-5','P-3','KC-10','E-3']; K=10
a,b=ev['all|CH'],ev['all|OCH']
rows=[f"\t\t{c:<6} & {a['P'][i]:.2f} & {a['R'][i]:.2f} & {a['F'][i]:.2f} & {a['support'][i]} & {b['P'][i]:.2f} & {b['R'][i]:.2f} & {b['F'][i]:.2f} & {b['support'][i]} \\\\" for i,c in enumerate(cl)]
rows.append(f"\t\t\\midrule\n\t\t\\textbf{{Accuracy}} & {a['acc']:.2f} & - & - & {a['n']} & {b['acc']:.2f} & - & - & {b['n']} \\\\")
open(O+'table_cls_rows10.tex','w').write('\n'.join(rows))
L=[]
for sn,lab in (('all','All processed images'),('unique','Exact duplicates removed'),('qc',r'Well-segmented subset ($\mathrm{sym}\geq 0.80$)')):
    r=[ev[f'{sn}|{fs}'] for fs in ('CH','OCH')]
    L.append(f"\t\t{lab} & {r[0]['n']} & {r[0]['acc']:.3f}$\\pm${r[0]['acc_sd']:.3f} & {r[1]['acc']:.3f}$\\pm${r[1]['acc_sd']:.3f} \\\\")
open(O+'table_subsets_rows10.tex','w').write('\n'.join(L))
fig,axs=plt.subplots(1,2,figsize=(14,6.2))
for ax,(r,t) in zip(axs,((a,'CH features'),(b,'OCH features'))):
    cm=r['CM']; ax.imshow(cm,cmap='Blues',vmin=0,vmax=100)
    for i in range(K):
        for j in range(K): ax.text(j,i,f'{cm[i,j]:.0f}',ha='center',va='center',color='white' if cm[i,j]>55 else 'black',fontsize=8)
    ax.set_xticks(range(K)); ax.set_yticks(range(K)); ax.set_xticklabels(cl,rotation=45,ha='right'); ax.set_yticklabels(cl); ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(f'SVM, {t} (acc={r["acc"]:.3f})',fontsize=10)
plt.tight_layout(); plt.savefig(O+'confusion_ch_och10.png',dpi=170)
print(open(O+'table_cls_rows10.tex').read()); print(open(O+'table_subsets_rows10.tex').read())
print('F1 CH',a['F'].round(2),'\nF1 OCH',b['F'].round(2),a['acc_sd'],b['acc_sd'])
