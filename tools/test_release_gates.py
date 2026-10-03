"""Check that private staging and missing licenses cannot pass publication gates."""
from pathlib import Path
import csv,hashlib,json,tempfile,unittest
from check_release import check

class ReleaseGateTests(unittest.TestCase):
    def fixture(self,folder,state):
        root=Path(folder)
        p=root/'RELEASE_STATUS.json'; p.write_text(json.dumps(state),encoding='utf-8')
        with (root/'RELEASE_MANIFEST.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=['file','sha256','bytes']); w.writeheader()
            w.writerow({'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
        return root
    def test_private_repository_is_not_public_release(self):
        with tempfile.TemporaryDirectory() as folder:
            root=self.fixture(folder,{'repository_url':'https://github.com/example/project','public_access_verified':False,'published':False})
            result=check(root)
            self.assertTrue(result['technical_checks_pass'])
            self.assertFalse(result['publication_ready'])
            self.assertIn('Public repository access not verified',result['open_release_gates'])
    def test_license_name_without_approved_text_is_insufficient(self):
        with tempfile.TemporaryDirectory() as folder:
            root=self.fixture(folder,{'software_license':'test-only','license_text_approved':True,'repository_url':'https://github.com/example/project','public_access_verified':True,'immutable_release_id':'test-only','published':True,'author_affirmations_complete':True})
            result=check(root)
            self.assertTrue(result['technical_checks_pass'])
            self.assertFalse(result['publication_ready'])
            self.assertIn('Software license/rights-holder approval pending',result['open_release_gates'])

if __name__=='__main__': unittest.main(verbosity=2)
