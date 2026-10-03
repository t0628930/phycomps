"""Check published arithmetic from component tables and raw timing measurements."""
from pathlib import Path
from collections import defaultdict
import argparse, csv, json, math, statistics
ROOT=Path(__file__).resolve().parents[1]

def rows(path):
    with path.open(newline='',encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))

def require(value,message):
    if not value: raise ValueError(message)

def close(a,b,label):
    a,b=float(a),float(b)
    require(math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12),label)

def validate(root):
    results=root/'validation_20261001/results'
    components=rows(results/'opi/opi_components_by_seed.csv')
    require(len(components)==150,'Expected 3 variants x 5 seeds x 2 threshold modes x 5 HD cutoffs')
    lookup={}
    for row in components:
        key=(row['variant'],int(row['seed']),row['thresholds'],float(row['hd_limit']))
        require(key not in lookup,'Duplicate component identity')
        vals=[float(row[k]) for k in ('range_compliance','thermo_compliance','smoothness')]
        require(all(math.isfinite(v) and 0<=v<=1 for v in vals),'Invalid compliance component')
        close(sum(a*b for a,b in zip(vals,(.5,.3,.2))),row['opi_original_weights'],'Original OPI mismatch')
        lookup[key]=vals
    actual_modes={x[2] for x in lookup}
    require(len(actual_modes)==2,'Expected two threshold definitions')
    grid=rows(results/'opi/opi_weight_grid.csv')
    require(len(grid)==660,'Expected 660 diagnostic settings')
    seen=set(); wins={'full':0,'iid':0}; ties={'full':0,'iid':0}
    for row in grid:
        weights=[float(row[k]) for k in ('w_range','w_thermo','w_smooth')]
        require(all(0<=w<=1 for w in weights),'Invalid simplex weights')
        close(sum(weights),1,'Weights do not sum to one')
        ints=tuple(round(w*10) for w in weights)
        require(all(math.isclose(w,n/10,abs_tol=1e-12) for w,n in zip(weights,ints)),'Weight off 0.1 grid')
        key=(row['thresholds'],float(row['hd_limit']),ints)
        require(key not in seen,'Duplicate weight-grid identity'); seen.add(key)
        require(key[0] in actual_modes and key[1] in (5,10,15,20,30),'Invalid threshold setting')
        means={}
        for label,variant in [('direct','non_residual_decoder'),('full','full'),('iid','prior_iid_uniform')]:
            means[label]=statistics.mean(sum(w*c for w,c in zip(weights,lookup[(variant,seed,key[0],key[1])])) for seed in range(42,47))
            close(means[label],row[label],'Grid mean mismatch: '+label)
        for baseline in ('full','iid'):
            delta=means['direct']-means[baseline]
            close(delta,row['direct_minus_'+baseline],'Grid difference mismatch')
            require(delta>=-1e-12,'Observed ranking reversal')
            if abs(delta)<=1e-12: ties[baseline]+=1
            else: wins[baseline]+=1
    require(wins=={'full':654,'iid':654} and ties=={'full':6,'iid':6},'Unexpected ranking counts')
    raw=rows(results/'benchmark_cpu/timings_raw.csv')
    require(len(raw)==700,'Expected 700 timing measurements')
    measurements=defaultdict(list); identity=set()
    expected_operations={'decoder_only_fixed_inputs','sampling_plus_decoder_fixed_noise','standalone_exact_box_audit','prior_nll_forward','prior_nll_plus_audit_shared_cache_forward'}
    for row in raw:
        key=(row['variant'],int(row['seed']),int(row['batch']),row['operation'])
        ident=key+(int(row['repeat']),)
        require(ident not in identity,'Duplicate timing identity'); identity.add(ident)
        require(key[0] in ('full','non_residual_decoder') and key[1] in range(42,47) and key[2] in (1,16) and key[3] in expected_operations and ident[-1] in range(7),'Invalid timing identity')
        value=float(row['milliseconds_per_batch'])
        require(math.isfinite(value) and value>0,'Invalid measured time')
        close(value/key[2],row['milliseconds_per_window_amortized'],'Amortized time mismatch')
        measurements[key].append(value)
    require(len(measurements)==100 and all(len(v)==7 for v in measurements.values()),'Incomplete checkpoint operations')
    checkpoints=rows(results/'benchmark_cpu/timings_by_checkpoint.csv')
    require(len(checkpoints)==100,'Expected 100 checkpoint summaries')
    seen=set()
    for row in checkpoints:
        key=(row['variant'],int(row['seed']),int(row['batch']),row['operation'])
        require(key not in seen,'Duplicate checkpoint summary'); seen.add(key)
        vals=measurements[key]
        for label,value in [('mean_ms_batch',statistics.mean(vals)),('median_ms_batch',statistics.median(vals)),('std_ms_batch',statistics.stdev(vals)),('min_ms_batch',min(vals)),('max_ms_batch',max(vals)),('mean_ms_window',statistics.mean(vals)/key[2])]:
            close(value,row[label],'Checkpoint statistic mismatch: '+label)
        require(int(row['repeats'])==7,'Incorrect repetition count')
    summaries=rows(results/'benchmark_cpu/timings_summary.csv')
    require(len(summaries)==20,'Expected 20 model/batch/operation summaries')
    seen=set()
    for row in summaries:
        key=(row['variant'],int(row['batch']),row['operation'])
        require(key not in seen,'Duplicate overall summary'); seen.add(key)
        vals=[statistics.mean(measurements[(key[0],seed,key[1],key[2])]) for seed in range(42,47)]
        close(statistics.mean(vals),row['mean_of_checkpoint_means_ms_batch'],'Overall mean mismatch')
        close(statistics.mean(vals)/key[1],row['mean_of_checkpoint_means_ms_window'],'Overall amortized mean mismatch')
        close(statistics.stdev(vals),row['between_checkpoint_std_ms_batch'],'Between-checkpoint spread mismatch')
        require(int(row['checkpoints'])==5,'Incorrect checkpoint count')
    return {'component_rows':150,'opi_settings_recomputed':660,'strict_advantages':wins,'ties':ties,'timed_calls_checked':700,'checkpoint_summaries_checked':100,'overall_timing_summaries_checked':20,'raw_sensor_data_required':False,'full_training_reproduced':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,default=ROOT)
    print(json.dumps(validate(parser.parse_args().root),indent=2))