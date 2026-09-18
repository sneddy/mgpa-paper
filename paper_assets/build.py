"""Rebuild the manuscript's reported tables and figures without model fitting."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = {'identity':'Identity','leace':'LEACE','mgpa_cf':'MGPA-CF',
          'mgpa_iter':'MGPA-Iter',
          'linear_only':'Linear-only','igbp':'IGBP','coral':'CORAL','featmap':'FEATMAP',
          'coral_0':r'CORAL $\to$ ref. 0','coral_1':r'CORAL $\to$ ref. 1',
          'featmap_0':r'FEATMAP $\to$ ref. 0','featmap_1':r'FEATMAP $\to$ ref. 1',
          'fit_mean':'Unconditional FIT mean','zero_target':'Zero score',
          'measurement':'MGPA-Iter'}
REUSE = ('identity','leace','mgpa_cf','mgpa_iter','coral','featmap')
TASKS = ('p300','n170','mmn')

def load(path):
    """Load a release-relative numerical artifact."""
    return json.loads((ROOT/path).read_text())

def number(value, digits=3):
    """Format an estimate without changing its stored precision."""
    return 'N/A' if value is None else f'{value:.{digits}f}'

def interval(stat, digits=3):
    """Format a signed paired contrast and percentile interval."""
    mean=stat.get('estimate',stat.get('mean'))
    lo,hi=stat['ci95'] if 'ci95' in stat else (stat['lower'],stat['upper'])
    return f'${mean:+.{digits}f}\\ [{lo:+.{digits}f}, {hi:+.{digits}f}]$'

def table(out, name, header, rows):
    """Write a compact TeX table and matching plain CSV."""
    columns='l'+'r'*(len(header)-1)
    lines=['% Generated from selected release results; no training or selection.',
           r'\begingroup\small\setlength{\tabcolsep}{4pt}',
           r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}'+columns+'@{}}',
           r'\toprule',' & '.join(header)+r' \\',r'\midrule']
    lines.extend(' & '.join(map(str,row))+r' \\' for row in rows)
    lines.extend([r'\bottomrule',r'\end{tabular*}\endgroup',''])
    (out/f'{name}.tex').write_text('\n'.join(lines))
    with (out/f'{name}.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(header);writer.writerows(rows)

def tables(out):
    """Render only tables included in the final manuscript."""
    v=load('paper_assets/published_values.json')
    methods=('identity','leace','mgpa_cf','coral_0','coral_1','featmap_0','featmap_1')
    rows=[]
    for m in methods:
        rows.append([LABELS[m],*[number(v['controlled_cf'][m][k]['mean']) for k in ('S','Tf','Tr','M')]])
    table(out,'controlled_cf_main',['Method',r'Source $\downarrow$',r'Frozen $\uparrow$',r'Refitted $\uparrow$','Movement'],rows)
    rows=[]
    for m in (*methods[:3],'mgpa_iter','linear_only','igbp',*methods[3:]):
        rows.append([LABELS[m],*[number(x) for x in v['real_native'][m]]])
    table(out,'real_native_main',['Method',*[f'{d} {k}' for d in ('N170','SSVEP') for k in ('S','Tf','Tr','M')]],rows)
    rows=[[('Conditional mean' if m=='mgpa_iter' else LABELS[m]),
           *[number(v['controlled_target_controls'][m][k]['mean'],4) for k in ('S','Tf','Tr','M')]]
          for m in ('mgpa_iter','fit_mean','zero_target')]
    table(out,'controlled_target_controls',['Target','Source','Frozen','Refitted','Movement'],rows)
    rows=[]
    for endpoint in ('budget_0.1','budget_0.25'):
        for m in ('measurement','igbp'):
            row=v['n170_igbp_budget'][f'{m}__{endpoint}']
            rows.append([endpoint.replace('budget_',''),LABELS[m],*[number(row[k]['mean'],4) for k in ('S','Tf','Tr','M')]])
    table(out,'n170_igbp_budget',['VAL budget','Method','Source','Frozen','Refitted','Movement'],rows)
    reuse=load('experiments/task_reuse/artifacts/tables/published.json')
    rows=[[LABELS[m],*[number(reuse['results'][t][m]['metrics'][metric]['estimate'])
                       for metric in ('worst','ordinary') for t in TASKS]] for m in REUSE]
    table(out,'reuse_main',['Method',*[f'{t.upper()} {metric}' for metric in ('worst','control') for t in TASKS]],rows)
    rows=[[LABELS[m],*[number(reuse['results'][t][m]['metrics'][metric]['estimate'])
                       for t in TASKS for metric in ('aligned','independent','reversed')]] for m in REUSE]
    table(out,'reuse_absolute_compact',['Method',*[f'{t.upper()} {a}' for t in TASKS for a in ('A','I','R')]],rows)
    rows=[]
    for t in TASKS:
        for m in ('mgpa_cf','mgpa_iter'):
            for ref in ('identity','leace'):
                c=reuse['contrasts'][t][m][ref]
                rows.append([t.upper(),LABELS[m],LABELS[ref],interval(c['worst']),interval(c['ordinary'],4)])
    table(out,'reuse_paired_ci',['Task','Method','Reference',r'$\Delta$ worst [95\% CI]',r'$\Delta$ control [95\% CI]'],rows)
    critic=load('experiments/temporal_n170/artifacts/tables/critic_comparison.json')
    rows=[[LABELS[m],number(r['L']['mean']),number(r['S']['mean'])] for m,r in critic['native'].items()]
    table(out,'n170_critic_compact',['Correction',r'$L\downarrow$',r'$S\downarrow$'],rows)
    with (out/'n170_critic_compact.tex').open('a') as stream:
        stream.write('\n'+r'\par\smallskip MGPA-Iter minus Linear-only (AUROC percentage points): '+
                     '; '.join(f'{k}: '+interval({z:([100*x for x in w] if isinstance(w,list) else 100*w)
                              for z,w in r.items() if z in ("mean","ci95")},2)
                              for k,r in critic['contrasts'].items())+'.\n')

def main():
    """Render to a separate directory, preserving the exact submitted assets."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'generated')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    tables(args.output)
    from render_gate import render as render_gate
    from render_reuse import render as render_reuse
    from render_theory import render as render_theory
    render_gate(args.output);render_reuse(args.output);render_theory(args.output)
    print(f'Rebuilt final tables and figures in {args.output.resolve()} (no fitting).')

if __name__=='__main__': main()
