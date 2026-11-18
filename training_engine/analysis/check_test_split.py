"""Re-implements dlorchestra.splitting.split_data (seed 42) on the 759 exam IDs."""
import csv, json, sys
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

p = pd.read_csv('segmentation/evaluation_c90z_384/predictions.csv')
scan = {}
for r in csv.DictReader(open('dataset/dataset.csv')):
    b, f = r.get('brand', '').strip(), r.get('field_strength_T', '').strip()
    scan[r['exam'].strip()] = f'{b}_{f}T' if b and f else 'unknown'
recs = [(lab, e) for lab in (0, 1) for e in sorted(p.loc[p.true_label == lab, 'exam_id'])]

def keys(strat, mn):
    labs = [str(l) for l, _ in recs]
    if not strat: return labs
    ks = [f"{l}_{scan.get(e.rstrip('ABCDEFGHIJKLMNOPQRSTUVWXYZ') or e, 'unknown')}" for l, e in recs]
    c = Counter(ks)
    return [k if c[k] >= mn else f'{l}_other' for k, l in zip(ks, labs)]

def split(strat, kfold):
    k = keys(strat, 5 if kfold else 2)
    dev, test, dk, _ = train_test_split(recs, k, test_size=0.15, random_state=42, stratify=k if all(k.count(x) >= 2 for x in set(k)) else None)
    print('  strat', strat, 'kfold', kfold, 'test stratified:', all(k.count(x) >= 2 for x in set(k)))
    if not kfold:
        tr, va = train_test_split(dev, test_size=0.2, random_state=42, stratify=dk if all(dk.count(x) >= 2 for x in set(dk)) else None)
        return {e for _, e in test}, [{e for _, e in va}]
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    return {e for _, e in test}, [{dev[i][1] for i in vi} for _, vi in skf.split(dev, dk)]

def actual(exp, fold=1):
    d = pd.read_csv(f'training_engine/experiments/{exp}/fold_{fold}/predictions.csv')
    return set(d.image.str.split('/').str[-1].str.replace(r'\.(nii\.gz|npy)$', '', regex=True))

r = {(s, k): split(s, k) for s in (False, True) for k in (False, True)}
print('n', len(recs), 'pos', sum(l for l, _ in recs))
print('label-only test: single==kfold', r[(False, False)][0] == r[(False, True)][0])
print('scanner test:    single==kfold', r[(True, False)][0] == r[(True, True)][0])
print('label-only vs scanner test overlap:', len(r[(False, False)][0] & r[(True, False)][0]), '/', len(r[(False, False)][0]))
print('sim matches MIP top90 single val:', actual('MIP_earlyfusion_top90') == r[(False, False)][1][0])
print('sim matches MIP top90 kfold val f1..5:', [actual('MIP_earlyfusion_top90_kfold', i + 1) == r[(False, True)][1][i] for i in range(5)])
print('sim matches 3D R3D18K_256 val:', actual('R3D18K_256_lr1e4_5439') == r[(True, False)][1][0])
json.dump({'label_only': sorted(r[(False, False)][0]), 'scanner': sorted(r[(True, False)][0])}, open(sys.argv[1], 'w'), indent=1)
