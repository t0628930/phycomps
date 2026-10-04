import argparse,csv,hashlib,json
from collections import Counter,defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean,median
p=argparse.ArgumentParser()
p.add_argument('--prepared',type=Path,required=True)
p.add_argument('--metrics',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
with a.prepared.open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
ts=[datetime.fromisoformat(x['timestamp']) for x in rows]
g=defaultdict(list)
for row,t in zip(rows,ts): g[row['segment_id']].append(t)
deltas=[(b-a).total_seconds() for a,b in zip(ts,ts[1:])]
inside=[deltas[i-1] for i in range(1,len(rows)) if rows[i]['segment_id']==rows[i-1]['segment_id']]
durations=[(max(t)-min(t)).total_seconds()/3600 for t in g.values()]
with a.metrics.open(encoding='utf-8',newline='') as f: m=list(csv.DictReader(f))
v={n:{r['seed']:float(r['operational_plausibility_index']) for r in m if r['variant']==n} for n in ['full','non_residual_decoder']}
assert v['full'].keys()==v['non_residual_decoder'].keys()
diff=mean(v['non_residual_decoder'][s]-v['full'][s] for s in v['full'])
r={'date':'2026-10-04','kind':'timestamp_and_rounding_recheck_no_retraining','prepared_sha256':hashlib.sha256(a.prepared.read_bytes()).hexdigest(),'metrics_sha256':hashlib.sha256(a.metrics.read_bytes()).hexdigest(),'rows':len(rows),'segments':len(g),'gaps_over_60_seconds':sum(x>60 for x in deltas),'first_timestamp':str(ts[0]),'last_timestamp':str(ts[-1]),'calendar_span_days':(ts[-1]-ts[0]).total_seconds()/86400,'within_segment_median_seconds':median(inside),'within_segment_mean_seconds':mean(inside),'within_segment_interval_counts':dict(sorted(Counter(inside).items())),'within_segment_duration_sum_hours':sum(durations),'mean_segment_duration_hours':mean(durations),'recorded_hour_00_to_06_rows':sum(t.hour<6 for t in ts),'opi_direct_mean':mean(v['non_residual_decoder'].values()),'opi_full_mean':mean(v['full'].values()),'opi_paired_difference':diff,'opi_paired_difference_4dp':format(diff,'.4f'),'limits':['Recorded timestamps only; timezone and interruption causes not inferred.','Segment span is last minus first timestamp, not measured installation uptime.','No retraining, resampling or metric-policy change.']}
assert r['rows']==50238 and r['segments']==62 and r['gaps_over_60_seconds']==61
assert r['opi_paired_difference_4dp']=='0.0109'
a.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
print(json.dumps(r,indent=2))