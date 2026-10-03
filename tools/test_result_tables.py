"""Negative controls for the table verifier, using temporary table copies only."""
from pathlib import Path
import csv,shutil,tempfile,unittest
from verify_result_tables import ROOT,validate

class ResultTableTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        for folder in ('opi','benchmark_cpu'):
            target=self.root/'validation_20261001/results'/folder; target.mkdir(parents=True)
            for source in (ROOT/'validation_20261001/results'/folder).glob('*.csv'):
                shutil.copy2(source,target/source.name)
    def tearDown(self): self.tmp.cleanup()
    def change(self,name,field,value):
        p=self.root/'validation_20261001/results'/name
        with p.open(newline='',encoding='utf-8') as f:
            reader=csv.DictReader(f); headers=reader.fieldnames; data=list(reader)
        data[0][field]=value
        with p.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=headers); w.writeheader(); w.writerows(data)
    def test_unchanged_passes(self):
        self.assertEqual(validate(self.root)['timed_calls_checked'],700)
    def test_changed_opi_detected(self):
        self.change('opi/opi_weight_grid.csv','direct','0.1')
        with self.assertRaises(ValueError): validate(self.root)
    def test_nonfinite_component_detected(self):
        self.change('opi/opi_components_by_seed.csv','range_compliance','nan')
        with self.assertRaises(ValueError): validate(self.root)
    def test_changed_timing_detected(self):
        self.change('benchmark_cpu/timings_raw.csv','milliseconds_per_batch','99999')
        with self.assertRaises(ValueError): validate(self.root)
    def test_changed_summary_detected(self):
        self.change('benchmark_cpu/timings_summary.csv','mean_of_checkpoint_means_ms_batch','99999')
        with self.assertRaises(ValueError): validate(self.root)

if __name__=='__main__': unittest.main(verbosity=2)