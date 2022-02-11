"""Unit tests for shabanipy.labber.labber_io module."""
import unittest
import warnings

from shabanipy.labber import LabberData


class TestWarnNotConstant(unittest.TestCase):
    def setUp(self):
        """Monkey patch LabberData to return synthetic data."""
        self.labberdata = LabberData("")
        self.labberdata.get_data = lambda *_: self.data

    def tearDown(self):
        del self.labberdata
        del self.data

    def assert_1_userwarning(self, warnings):
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].category, UserWarning)

    def test_default_1_percent(self):
        with warnings.catch_warnings(record=True) as w:
            self.data = [99] * 100 + [100]
            self.labberdata.warn_not_constant("")
            self.assertEqual(len(w), 0)
            self.data += [99]
            self.labberdata.warn_not_constant("")
            self.assert_1_userwarning(w)

    def test_user_specified_threshold(self):
        self.data = [1, 2]
        with warnings.catch_warnings(record=True) as w:
            self.labberdata.warn_not_constant("", 0.51)
            self.assertEqual(len(w), 0)
            self.labberdata.warn_not_constant("", 0.49)
            self.assert_1_userwarning(w)
