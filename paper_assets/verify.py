"""Recompute published statistics from retained participant cells, without fitting."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mgpa.audit.reporting import ensemble_summary, metric_summary, paired_contrast


def load(path):
    """Read a release-relative JSON record."""
    return json.loads((ROOT/path).read_text())


def check(actual,expected,name,checks):
    """Require full-precision agreement, not merely identical printed rounding."""
    error=float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
    if error>1e-12:
        raise AssertionError(f'{name}: maximum discrepancy {error:g}')
    checks[name]=error


def cells(protocol):
    """Group complete cells by comparison, method and endpoint."""
    result=defaultdict(list)
    for path in sorted((ROOT/f'experiments/{protocol}/artifacts/tables/cells').rglob('*.json')):
        c=json.loads(path.read_text())
        result[path.parent.name,c['job']['method'],c['endpoint']].append(c)
    return result


def compare_stat(actual,expected,name,checks):
    """Check estimate and percentile limits with the same participant draws."""
    check([actual['mean'],*actual['ci95']],[expected['mean'],*expected['ci95']],name,checks)


def verify_manifest():
    """Authenticate the released code, configurations and frozen numerical cells."""
    manifest=load('release_manifest.json')
    files=manifest['numerical_files'] | manifest['implementation_files']
    files.update({p['configuration']['path']:p['configuration']['sha256']
                  for p in manifest['protocols'].values()})
    for name,expected in files.items():
        path=ROOT/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise AssertionError(f'Released snapshot changed: {name}')
    print(f'PASS: {len(files)} released files match the manifest.')


def main():
    """Verify source maxima, tasks, movement and paired contrasts against the paper."""
    verify_manifest()
    values=load('paper_assets/published_values.json');checks={}
    metrics=('S','Tf','Tr','M')
    controlled=cells('controlled')
    for (group,method,endpoint),records in controlled.items():
        if group=='closed_form':
            expected=values['controlled_cf'][method]
        elif group=='target':
            expected=values['controlled_target_controls'][method]
        elif group=='gate' and not method.startswith('random_'):
            expected=values['controlled_mechanisms']['gate'][method]['metrics']
        else:
            continue
        for metric in metrics:
            compare_stat(metric_summary(records,metric),expected[metric],f'controlled/{group}/{method}/{metric}',checks)
    gates=values['controlled_mechanisms']['gate']
    reference=controlled['gate','identity','native']
    for method in ('measurement','pca','ungated','random'):
        groups=([controlled['gate',f'random_{i}','budget_0.05'] for i in range(5)] if method=='random'
                else [controlled['gate',method,'budget_0.05']])
        actual=ensemble_summary(groups,'Tf',reference_cells=reference)
        compare_stat(actual,gates[method]['delta_Tf'],f'gate/{method}/paired_delta_Tf',checks)
        if method=='random':
            for metric in metrics:
                compare_stat(ensemble_summary(groups,metric),gates[method]['metrics'][metric],f'gate/random/{metric}',checks)
    for protocol,offset in [('temporal_n170',0),('recorded_ssvep',4)]:
        grouped=cells(protocol)
        for (group,method,endpoint),records in grouped.items():
            if endpoint!='native':continue
            expected=values['real_native'][method][offset:offset+4]
            actual=[metric_summary(records,m,resamples=1)['mean'] for m in metrics]
            check(actual,expected,f'{protocol}/{method}/native',checks)
        if protocol=='temporal_n170':
            for method,group in [('igbp','igbp'),('measurement','matched_budget')]:
                for endpoint in ('budget_0.1','budget_0.25'):
                    expected=values['n170_igbp_budget'][f'{method}__{endpoint}']
                    for metric in metrics:
                        actual=metric_summary(grouped[group,method,endpoint],metric)
                        compare_stat(actual,expected[metric],f'n170/{method}/{endpoint}/{metric}',checks)
            critic=load('experiments/temporal_n170/artifacts/tables/critic_comparison.json')
            for method in ('identity','mgpa_iter','linear_only'):
                for metric,families in [('S',None),('L',['linear','whitened_linear'])]:
                    actual=metric_summary(grouped['native',method,'native'],'S',source_families=families)
                    compare_stat(actual,critic['native'][method][metric],f'n170/critic/{method}/{metric}',checks)
            for metric,expected in critic['contrasts'].items():
                actual=paired_contrast(grouped['native','mgpa_iter','native'],grouped['native','linear_only','native'],metric)
                compare_stat(actual,expected,f'n170/critic/difference/{metric}',checks)
    asset_hashes=load('release_manifest.json')['paper_assets']
    for filename,metadata in asset_hashes.items():
        actual=hashlib.sha256((ROOT/'paper_assets'/filename).read_bytes()).hexdigest()
        if actual!=metadata['sha256']:raise AssertionError(f'Original paper asset changed: {filename}')
    result={'status':'PASS','checked_statistics':len(checks),'maximum_absolute_error':max(checks.values()),
            'checks':checks,'original_paper_assets_checked':len(asset_hashes),'bootstrap_resamples':20000,
            'model_fitting':False,'reader_fitting':False,'endpoint_selection':False,
            'scope':'Participant statistics; reuse exact quadratic bootstrap is verified by its standalone report.'}
    (ROOT/'paper_assets/verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f"PASS: {len(checks)} full-precision statistics; maximum error {max(checks.values()):.3g}.")

if __name__=='__main__': main()
