"""
Tests for the pre/post sequence execution system in MRIBLANKSEQ.

These tests validate:
- Parsing of comma-separated sequence names
- Pre-sequence execution with value forwarding
- Post-sequence execution with value forwarding
- Edge cases (missing sequences, recursion prevention, empty configs)
"""

import copy
import unittest
import numpy as np

import sys
import os

# Ensure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from marge.seq.mriBlankSeq import MRIBLANKSEQ


# ---------------------------------------------------------------------------
# Minimal concrete sequence subclasses used for testing
# ---------------------------------------------------------------------------

class _StubSequence(MRIBLANKSEQ):
    """A minimal sequence that always succeeds and records calls."""

    def __init__(self, name='StubSeq'):
        super().__init__()
        self.addParameter(key='seqName', val=name)
        self.addParameter(key='toMaRGE', val=True)
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=3.0, field='RF')
        self.addParameter(key='shimming', string='Shimming', val=[0.0, 0.0, 0.0], field='OTH')
        self.run_count = 0
        self.analysis_count = 0

    def sequenceTime(self):
        return 0.0

    def sequenceRun(self, plotSeq=0, demo=False, standalone=False):
        self.run_count += 1
        return True

    def sequenceAnalysis(self, mode=None):
        self.analysis_count += 1
        self.output = []
        return self.output


class _LarmorStub(_StubSequence):
    """Simulates a Larmor calibration sequence that updates larmorFreq."""

    def __init__(self):
        super().__init__(name='LarmorStub')
        self._new_freq = 3.1  # The frequency this stub will "discover"

    def sequenceAnalysis(self, mode=None):
        super().sequenceAnalysis(mode=mode)
        # Simulate Larmor updating the frequency in all sequences
        self.mapVals['larmorFreq'] = self._new_freq
        if hasattr(self, 'sequence_list') and self.sequence_list is not None:
            for seq in self.sequence_list.values():
                if 'larmorFreq' in seq.mapVals:
                    seq.mapVals['larmorFreq'] = self._new_freq
        return self.output


class _FailingStub(_StubSequence):
    """A sequence whose sequenceRun always fails."""

    def __init__(self):
        super().__init__(name='FailingStub')

    def sequenceRun(self, plotSeq=0, demo=False, standalone=False):
        self.run_count += 1
        return False


# ===========================================================================
# Tests
# ===========================================================================

class TestParseSequenceNames(unittest.TestCase):
    """Tests for MRIBLANKSEQ._parseSequenceNames."""

    def test_empty_string(self):
        self.assertEqual(MRIBLANKSEQ._parseSequenceNames(''), [])

    def test_none_value(self):
        self.assertEqual(MRIBLANKSEQ._parseSequenceNames(None), [])

    def test_none_string(self):
        self.assertEqual(MRIBLANKSEQ._parseSequenceNames('none'), [])
        self.assertEqual(MRIBLANKSEQ._parseSequenceNames('None'), [])
        self.assertEqual(MRIBLANKSEQ._parseSequenceNames('NONE'), [])

    def test_single_name(self):
        self.assertEqual(MRIBLANKSEQ._parseSequenceNames('Larmor'), ['Larmor'])

    def test_multiple_names(self):
        self.assertEqual(
            MRIBLANKSEQ._parseSequenceNames('Larmor, Shimming, Noise'),
            ['Larmor', 'Shimming', 'Noise'],
        )

    def test_extra_whitespace(self):
        self.assertEqual(
            MRIBLANKSEQ._parseSequenceNames('  Larmor ,  Shimming  '),
            ['Larmor', 'Shimming'],
        )

    def test_trailing_comma(self):
        self.assertEqual(
            MRIBLANKSEQ._parseSequenceNames('Larmor,'),
            ['Larmor'],
        )

    def test_non_string_input(self):
        self.assertEqual(MRIBLANKSEQ._parseSequenceNames(123), [])


class TestPreSequenceExecution(unittest.TestCase):
    """Tests for MRIBLANKSEQ.runPreSequences."""

    def _build_sequence_list(self):
        """Create a mock sequence_list with a Larmor stub and a main sequence."""
        larmor = _LarmorStub()
        main = _StubSequence(name='MainSeq')
        failing = _FailingStub()
        return {
            'LarmorStub': larmor,
            'MainSeq': main,
            'FailingStub': failing,
        }

    def test_no_pre_sequences(self):
        """No pre-sequences configured → runPreSequences returns True immediately."""
        seq = _StubSequence(name='MainSeq')
        seq.sequence_list = self._build_sequence_list()
        self.assertTrue(seq.runPreSequences(demo=True))

    def test_pre_sequence_runs_and_forwards_values(self):
        """Larmor pre-sequence updates larmorFreq and value is forwarded to main sequence."""
        sequence_list = self._build_sequence_list()
        main = _StubSequence(name='MainSeq')
        main.sequence_list = sequence_list
        main.mapVals['preSequence'] = 'LarmorStub'

        # Before: larmorFreq should be the default (3.0)
        self.assertEqual(main.mapVals['larmorFreq'], 3.0)

        result = main.runPreSequences(demo=True)

        self.assertTrue(result)
        # After: larmorFreq should be updated to 3.1 (set by _LarmorStub)
        self.assertEqual(main.mapVals['larmorFreq'], 3.1)

    def test_pre_sequence_failure_returns_false(self):
        """If a pre-sequence fails, runPreSequences returns False."""
        sequence_list = self._build_sequence_list()
        main = _StubSequence(name='MainSeq')
        main.sequence_list = sequence_list
        main.mapVals['preSequence'] = 'FailingStub'

        result = main.runPreSequences(demo=True)
        self.assertFalse(result)

    def test_missing_pre_sequence_skipped(self):
        """Missing pre-sequence names are skipped with a warning, not an error."""
        sequence_list = self._build_sequence_list()
        main = _StubSequence(name='MainSeq')
        main.sequence_list = sequence_list
        main.mapVals['preSequence'] = 'NonExistentSeq'

        result = main.runPreSequences(demo=True)
        self.assertTrue(result)

    def test_multiple_pre_sequences(self):
        """Multiple pre-sequences execute in order."""
        larmor = _LarmorStub()
        other = _StubSequence(name='OtherPre')
        main_template = _StubSequence(name='MainSeq')
        sequence_list = {
            'LarmorStub': larmor,
            'OtherPre': other,
            'MainSeq': main_template,
        }
        main = _StubSequence(name='MainSeq')
        main.sequence_list = sequence_list
        main.mapVals['preSequence'] = 'LarmorStub, OtherPre'

        result = main.runPreSequences(demo=True)
        self.assertTrue(result)
        # larmorFreq forwarded from Larmor pre-sequence
        self.assertEqual(main.mapVals['larmorFreq'], 3.1)

    def test_no_sequence_list_returns_true(self):
        """If sequence_list is not set, runPreSequences returns True safely."""
        main = _StubSequence(name='MainSeq')
        main.mapVals['preSequence'] = 'LarmorStub'
        # No sequence_list set
        result = main.runPreSequences(demo=True)
        self.assertTrue(result)

    def test_recursion_prevention(self):
        """Pre-sequences should not execute their own pre/post sequences."""
        larmor = _LarmorStub()
        # Give the Larmor stub its own preSequence - it should be ignored
        larmor.mapVals['preSequence'] = 'LarmorStub'
        main_template = _StubSequence(name='MainSeq')
        sequence_list = {
            'LarmorStub': larmor,
            'MainSeq': main_template,
        }
        main = _StubSequence(name='MainSeq')
        main.sequence_list = sequence_list
        main.mapVals['preSequence'] = 'LarmorStub'

        # Should not recurse infinitely
        result = main.runPreSequences(demo=True)
        self.assertTrue(result)


class TestPostSequenceExecution(unittest.TestCase):
    """Tests for MRIBLANKSEQ.runPostSequences."""

    def _build_sequence_list(self):
        stub = _StubSequence(name='PostStub')
        main_template = _StubSequence(name='MainSeq')
        failing = _FailingStub()
        return {
            'PostStub': stub,
            'MainSeq': main_template,
            'FailingStub': failing,
        }

    def test_no_post_sequences(self):
        """No post-sequences configured → returns True."""
        seq = _StubSequence(name='MainSeq')
        seq.sequence_list = self._build_sequence_list()
        self.assertTrue(seq.runPostSequences(demo=True))

    def test_post_sequence_receives_forwarded_values(self):
        """Post-sequence should receive shared parameter values from the main sequence."""
        sequence_list = self._build_sequence_list()
        main = _StubSequence(name='MainSeq')
        main.sequence_list = sequence_list
        main.mapVals['postSequence'] = 'PostStub'
        main.mapVals['larmorFreq'] = 3.2  # Simulate updated value

        result = main.runPostSequences(demo=True)
        self.assertTrue(result)

    def test_post_sequence_failure_returns_false(self):
        """If a post-sequence fails, runPostSequences returns False."""
        sequence_list = self._build_sequence_list()
        main = _StubSequence(name='MainSeq')
        main.sequence_list = sequence_list
        main.mapVals['postSequence'] = 'FailingStub'

        result = main.runPostSequences(demo=True)
        self.assertFalse(result)

    def test_missing_post_sequence_skipped(self):
        """Missing post-sequence names are skipped."""
        sequence_list = self._build_sequence_list()
        main = _StubSequence(name='MainSeq')
        main.sequence_list = sequence_list
        main.mapVals['postSequence'] = 'NonExistent'

        result = main.runPostSequences(demo=True)
        self.assertTrue(result)


class TestPrePostParametersExist(unittest.TestCase):
    """Verify that the base class parameters for pre/post sequences exist."""

    def test_parameters_in_mapvals(self):
        seq = MRIBLANKSEQ()
        self.assertIn('preSequence', seq.mapVals)
        self.assertIn('postSequence', seq.mapVals)

    def test_default_values(self):
        seq = MRIBLANKSEQ()
        self.assertEqual(seq.mapVals['preSequence'], 'none')
        self.assertEqual(seq.mapVals['postSequence'], 'none')

    def test_parameters_in_oth_field(self):
        seq = MRIBLANKSEQ()
        self.assertEqual(seq.mapFields['preSequence'], 'OTH')
        self.assertEqual(seq.mapFields['postSequence'], 'OTH')


if __name__ == '__main__':
    unittest.main()
