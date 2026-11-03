"""Area-weighting and turn-away retention regressions for the 90% gate."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from falhv2.tests.test_depth_replacement import frame, grid, prediction
from falhv2.map_maintainers.depth_replace import DepthReplacementMap, coverage_ratio

BOX = ((-.5, -.4, 2.), (.5, .4, 3.))  # world XY: [2,3] x [-.5,.5]


class CoverageRatioTests(unittest.TestCase):
    def small_grid(self):
        return replace(grid(), resolution=1., origin=(0.,0.), rows=('....',)*4)

    def test_partial_cells_use_area_and_grid_row_orientation(self):
        g=self.small_grid();mask=np.zeros((4,4),bool)
        mask[3,0]=True
        # Width 2; only .75 of the leftmost cell overlaps. Height cancels.
        self.assertAlmostEqual(coverage_ratio((.25,.2,0),(2.25,.8,1),g,mask),.375)
        self.assertEqual(coverage_ratio((.25,3.2,0),(2.25,3.8,1),g,mask),0.)
        # A boundary touch has zero area.
        self.assertEqual(coverage_ratio((1.,0.,0),(2.,1.,1),g,mask),0.)

    def test_outside_grid_area_remains_in_denominator(self):
        g=self.small_grid();mask=np.ones((4,4),bool)
        self.assertAlmostEqual(coverage_ratio((-9.,0.,0),(1.,1.,1),g,mask),.1)
        self.assertEqual(coverage_ratio((-2.,0.,0),(-1.,1.,1),g,mask),0.)
        self.assertEqual(coverage_ratio((.1,.2,0),(.7,.8,1),g,mask),1.)

    def test_height_is_ignored_and_invalid_aabb_rejected(self):
        g=self.small_grid();mask=np.ones((4,4),bool)
        self.assertEqual(coverage_ratio((0,0,-5),(2,2,-4),g,mask),1.)
        with self.assertRaises(ValueError):
            coverage_ratio((0,0,0),(0,2,1),g,mask)

    def builder(self):
        b=DepthReplacementMap();b.set_grid(grid())
        f=frame();b.integrate(f,prediction(f,(BOX,)))
        self.assertEqual(len(b.objects),1)
        return b

    def test_partial_overlap_preserves_old_and_rejects_new(self):
        b=self.builder();old=b.objects[0]
        mask=np.ones((80,100),bool);mask[:,30:32]=False  # 80% of box covered
        with patch('falhv2.map_maintainers.depth_replace.depth_coverage',return_value=mask):
            for stamp in range(2,5):
                f=frame(stamp);u=b.integrate(f,prediction(f,(BOX,)))
                self.assertEqual(b.objects,(old,))
                self.assertAlmostEqual(u['decisions'][0]['coverage_ratio'],.8)
                self.assertEqual(u['removed_ids'],[])
                self.assertEqual(u['coverage_filtered_indices'],[0])
                self.assertEqual(u['added_ids'],[])

    def test_exact_90_percent_replaces_and_accepts(self):
        b=self.builder();old=b.objects[0]
        mask=np.ones((80,100),bool);mask[:,30]=False
        f=frame(2)
        with patch('falhv2.map_maintainers.depth_replace.depth_coverage',return_value=mask):
            u=b.integrate(f,prediction(f,(BOX,)))
        self.assertAlmostEqual(u['decisions'][0]['coverage_ratio'],.9)
        self.assertEqual(u['removed_ids'],[old.object_id])
        self.assertEqual(u['added_ids'],['observation-2-0'])
        self.assertEqual(u['coverage_filtered_indices'],[])

    def test_config_and_saved_policy(self):
        for v in [0,-1,1.1,float('nan'),float('inf'),True]:
            with self.subTest(v=v),self.assertRaises(ValueError):
                DepthReplacementMap(min_coverage_ratio=v)
        b=self.builder()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'map.json';b.save(p)
            restored=DepthReplacementMap.load(p)
            self.assertEqual(restored.min_coverage_ratio,.9)
            self.assertEqual(restored.to_scene(),b.to_scene())
            import json
            old=b.to_scene();old['mapping'].update(schema_version=2,policy='xy_depth_replacement')
            p.write_text(json.dumps(old))
            with self.assertRaisesRegex(ValueError,'fresh map'):
                DepthReplacementMap.load(p)
