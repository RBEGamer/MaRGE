"""
Created on Wed Mar 11 2026
@author: OpenAI Codex
@Summary: Play a MIDI file through the gradient channels as audible square waves.
"""

import os
import struct
from bisect import bisect_right
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

import marge.configs.hw_config as hw
import marge.configs.units as units
import marge.controller.experiment_gui as ex
import marge.seq.mriBlankSeq as blankSeq
from marge.marcos.marcos_client.local_config import fpga_clk_freq_MHz
from marge.utils.marcos_runtime import active_grad_board


@dataclass
class _MidiNoteInterval:
    start_tick: int
    end_tick: int
    note: int
    velocity: int
    track: int
    channel: int
    start_s: float = 0.0
    end_s: float = 0.0


class MidiGradient(blankSeq.MRIBLANKSEQ):
    _DEFAULT_TEMPO_US_PER_QUARTER = 500_000
    _MIN_EVENT_SPACING_US = 2.0
    _MAX_BATCH_SPAN_TICKS = int(np.iinfo(np.int32).max - 4096)

    def __init__(self):
        super(MidiGradient, self).__init__()
        self.output = []
        self._cached_midi_key = None
        self._cached_intervals = None
        self._selected_voice_info = []

        self.addParameter(key='seqName', string='MidiGradient', val='MidiGradient')
        self.addParameter(key='toMaRGE', val=True)
        self.addParameter(
            key='midiFile',
            string='MIDI file',
            val='',
            field='IM',
            tip='Absolute or workspace-relative path to a .mid/.midi file.',
        )
        self.addParameter(
            key='playbackAxis',
            string='Playback axis',
            val='all',
            field='IM',
            tip="'x', 'y', 'z' for one selected axis or 'all' for x/y/z playback.",
        )
        self.addParameter(
            key='singleVoice',
            string='Single voice index',
            val=1,
            field='IM',
            tip='1-based extracted monophonic voice index used when playback axis is x, y or z.',
        )
        self.addParameter(
            key='gradientAmplitude',
            string='Gradient amplitude (a.u.)',
            val=0.1,
            field='IM',
            tip='Normalized square-wave amplitude in the range [0, 1].',
        )
        self.addParameter(
            key='startDelay',
            string='Start delay (ms)',
            val=50.0,
            units=units.ms,
            field='IM',
        )
        self.addParameter(
            key='tempoScale',
            string='Tempo scale',
            val=1.0,
            field='IM',
            tip='Playback speed multiplier. Values > 1 are faster.',
        )
        self.addParameter(
            key='transpose',
            string='Transpose (semitones)',
            val=0,
            field='IM',
            tip='Signed semitone shift applied before converting notes to frequency.',
        )
        self.addParameter(
            key='maxFrequency',
            string='Max frequency (Hz)',
            val=2000.0,
            field='IM',
            tip='High notes are clipped to this value to keep gradient updates bounded.',
        )
        self.addParameter(
            key='gradUpdateRate',
            string='Grad update rate (MSPS)',
            val=0.1,
            field='IM',
            tip='Gradient serializer rate used when creating the MaRCoS experiment.',
        )
        self.addParameter(
            key='includePercussion',
            string='Include percussion',
            val=0,
            field='IM',
            tip='0 ignores MIDI channel 10, 1 converts it into pitched playback.',
        )
        self.addParameter(
            key='shimming',
            string='Shimming',
            val=[0.0, 0.0, 0.0],
            units=units.sh,
            field='IM',
        )
        self._default_map_vals = deepcopy(self.mapVals)

    def sequenceInfo(self):
        print("MidiGradient")
        print("Play a MIDI file as audible square waves on the gradient channels.")
        print("Playback axis x, y or z extracts one monophonic voice onto that axis.")
        print("Playback axis all extracts up to three voices and maps them to X, Y and Z.\n")

    @classmethod
    def _hardware_min_event_spacing_us(cls, grad_update_rate):
        return max(cls._MIN_EVENT_SPACING_US, 1.0 / float(grad_update_rate))

    @classmethod
    def _max_batch_span_us(cls):
        return cls._MAX_BATCH_SPAN_TICKS / fpga_clk_freq_MHz

    def _restore_default_mapvals(self):
        for key, default_value in self._default_map_vals.items():
            if key not in self.mapVals:
                self.mapVals[key] = deepcopy(default_value)

        legacy_mode = self.mapVals.get('playbackMode', '')
        legacy_axis = self.mapVals.get('singleAxis', '')
        if 'playbackAxis' not in self.mapVals or self.mapVals['playbackAxis'] in ('', None):
            if str(legacy_mode).strip().lower() == 'triple':
                self.mapVals['playbackAxis'] = 'all'
            elif str(legacy_axis).strip().lower() in ('x', 'y', 'z'):
                self.mapVals['playbackAxis'] = str(legacy_axis).strip().lower()
            else:
                self.mapVals['playbackAxis'] = deepcopy(self._default_map_vals['playbackAxis'])

    def loadParams(self, directory='experiments/parameterization', file=None):
        super(MidiGradient, self).loadParams(directory=directory, file=file)
        self._restore_default_mapvals()

    def sequenceTime(self):
        try:
            _, intervals = self._get_midi_intervals(
                self.mapVals['midiFile'],
                bool(self.mapVals['includePercussion']),
            )
        except Exception:
            return 0

        if not intervals:
            return 0

        tempo_scale = float(self.mapVals['tempoScale'])
        if tempo_scale <= 0:
            return 0

        duration_s = max(interval.end_s for interval in intervals) / tempo_scale
        return duration_s / 60.0

    def sequenceRun(self, plotSeq=0, demo=False, standalone=False):
        self.demo = demo
        self.plotSeq = bool(plotSeq)
        self.standalone = standalone
        self.output = []
        self._selected_voice_info = []
        self._playback_batches = []

        try:
            midi_path, base_intervals = self._get_midi_intervals(self.midiFile, bool(self.includePercussion))
        except Exception as exc:
            print("ERROR: Unable to load MIDI file.")
            print(exc)
            return False

        if not base_intervals:
            print("ERROR: No note events found in the selected MIDI file.")
            return False

        axis_name = str(self.playbackAxis).strip().lower()
        if axis_name not in ('x', 'y', 'z', 'all'):
            print("ERROR: Playback axis must be 'x', 'y', 'z' or 'all'.")
            return False

        amplitude = float(self.gradientAmplitude)
        if amplitude < 0 or amplitude > 1:
            print("ERROR: Gradient amplitude must be in the range [0, 1].")
            return False

        tempo_scale = float(self.tempoScale)
        if tempo_scale <= 0:
            print("ERROR: Tempo scale must be positive.")
            return False

        max_frequency = float(self.maxFrequency)
        if max_frequency <= 0:
            print("ERROR: Max frequency must be positive.")
            return False

        grad_update_rate = float(self.gradUpdateRate)
        if grad_update_rate <= 0:
            print("ERROR: Gradient update rate must be positive.")
            return False
        min_event_spacing_us = self._hardware_min_event_spacing_us(grad_update_rate)

        intervals = [
            _MidiNoteInterval(
                start_tick=interval.start_tick,
                end_tick=interval.end_tick,
                note=interval.note,
                velocity=interval.velocity,
                track=interval.track,
                channel=interval.channel,
                start_s=interval.start_s / tempo_scale,
                end_s=interval.end_s / tempo_scale,
            )
            for interval in base_intervals
        ]
        voices = self._partition_voices(intervals)
        if not voices:
            print("ERROR: No playable voices could be extracted from the MIDI file.")
            return False

        selected_voices = self._select_voices(voices, axis_name)
        if not selected_voices:
            print("ERROR: No voice selected for playback.")
            return False

        transpose = int(self.transpose)
        clock_period_us = 1.0 / fpga_clk_freq_MHz
        init_time_us = float(self.startDelay) * 1e6
        playback_start_us = init_time_us + min_event_spacing_us + clock_period_us
        axis_waveforms = {0: (np.array([], dtype=float), np.array([], dtype=float)),
                          1: (np.array([], dtype=float), np.array([], dtype=float)),
                          2: (np.array([], dtype=float), np.array([], dtype=float))}

        for voice_info in selected_voices:
            axis_index = voice_info['axis_index']
            times, values = self._build_voice_waveform(
                intervals=voice_info['intervals'],
                amplitude=amplitude,
                transpose=transpose,
                max_frequency=max_frequency,
                playback_start_us=playback_start_us + axis_index * min_event_spacing_us,
                shimming=float(self.shimming[axis_index]),
                min_event_spacing_us=min_event_spacing_us,
            )
            if times.size == 0:
                continue

            axis_waveforms[axis_index] = (times, values)
            voice_info['wave_times_ms'], voice_info['wave_freq_hz'] = self._voice_plot_data(
                voice_info['intervals'],
                transpose=transpose,
                max_frequency=max_frequency,
            )

        axis_waveforms, n_retimed_events = self._retime_axis_waveforms_for_hardware(
            axis_waveforms=axis_waveforms,
            min_event_spacing_us=min_event_spacing_us,
        )

        playback_batches = self._split_waveforms_into_batches(
            axis_waveforms=axis_waveforms,
            initial_grad_state=np.array(self.shimming, dtype=float),
            min_event_spacing_us=min_event_spacing_us,
        )
        if not playback_batches:
            print("ERROR: No gradient playback batches could be created from the MIDI file.")
            return False

        self._playback_batches = playback_batches
        self.flo_dict = playback_batches[0]['flo_dict']
        if len(playback_batches) > 1:
            print(
                f"INFO: MIDI playback split into {len(playback_batches)} batches "
                f"to stay within the hardware/compiler limits "
                f"({hw.maxOrders} events, {self._max_batch_span_us() * 1e-3:.1f} ms per batch)."
            )

        self._selected_voice_info = selected_voices
        self.raw_data_name = f"{self.mapVals['seqName']}_{os.path.splitext(os.path.basename(midi_path))[0]}"
        self.mapVals['midiFileResolved'] = midi_path
        self.mapVals['nMidiNotes'] = len(intervals)
        self.mapVals['nExtractedVoices'] = len(voices)
        self.mapVals['nSelectedVoices'] = len(selected_voices)
        self.mapVals['playedAxes'] = [info['axis_name'] for info in selected_voices]
        self.mapVals['playedVoiceIndices'] = [info['voice_index'] + 1 for info in selected_voices]
        self.mapVals['midiDurationMs'] = max(interval.end_s for interval in intervals) * 1e3
        self.mapVals['nPlaybackBatches'] = len(playback_batches)
        self.mapVals['batchEventCounts'] = [batch['n_orders'] for batch in playback_batches]
        self.mapVals['batchDurationsMs'] = [batch['duration_us'] * 1e-3 for batch in playback_batches]
        self.mapVals['nRetimedGradientEvents'] = int(n_retimed_events)

        if self.demo:
            if self.plotSeq and len(playback_batches) > 1:
                print("WARNING: Plotting only the first playback batch.")
            return True

        grad_kwargs = {
            'rx_t': 0.5,
            'grad_max_update_rate': grad_update_rate,
            'auto_leds': False,
            'init_gpa': False,
        }
        if active_grad_board() == 'gpa-fhdo':
            grad_kwargs['gpa_fhdo_offset_time'] = 10

        if self.plotSeq and self.standalone:
            self.expt = ex.Experiment(**grad_kwargs)
            if not self.floDict2Exp():
                print("ERROR: Sequence waveforms out of hardware bounds.")
                self.expt.__del__()
                return False
            if len(playback_batches) > 1:
                print("WARNING: Plotting only the first playback batch.")
            self.sequencePlot(standalone=True)
            self.expt.__del__()
            return True

        if self.plotSeq:
            self.expt = ex.Experiment(**grad_kwargs)
            if not self.floDict2Exp():
                print("ERROR: Sequence waveforms out of hardware bounds.")
                self.expt.__del__()
                return False
            if len(playback_batches) > 1:
                print("WARNING: Plotting only the first playback batch.")
            self.expt.__del__()
            return True

        for batch_index, batch in enumerate(playback_batches, start=1):
            print(f"Playback batch {batch_index}/{len(playback_batches)} running...")
            self.flo_dict = batch['flo_dict']
            self.expt = ex.Experiment(**grad_kwargs)
            try:
                if not self.floDict2Exp():
                    print("ERROR: Sequence waveforms out of hardware bounds.")
                    return False
                self.expt.run()
            finally:
                self.expt.__del__()
            print(f"Playback batch {batch_index}/{len(playback_batches)} ready!")

        self.flo_dict = playback_batches[0]['flo_dict']

        return True

    def sequenceAnalysis(self, mode=None):
        self.mode = mode
        x_data = []
        y_data = []
        legend = []

        for info in self._selected_voice_info:
            if len(info.get('wave_times_ms', [])) == 0:
                continue
            x_data.append(np.array(info['wave_times_ms'], dtype=float))
            y_data.append(np.array(info['wave_freq_hz'], dtype=float))
            legend.append(f"{info['axis_name'].upper()} / voice {info['voice_index'] + 1}")

        if x_data:
            self.output = [{
                'widget': 'curve',
                'xData': x_data,
                'yData': y_data,
                'legend': legend,
                'xLabel': 'Time (ms)',
                'yLabel': 'Frequency (Hz)',
                'title': 'Gradient MIDI playback',
                'row': 0,
                'col': 0,
            }]
        else:
            self.output = []

        self.saveRawData()

        if mode == 'Standalone' and self.output:
            self.plotResults()

        return self.output

    @classmethod
    def _read_vlq(cls, data, offset):
        value = 0
        while True:
            if offset >= len(data):
                raise ValueError("Unexpected end of MIDI file while reading VLQ.")
            byte = data[offset]
            offset += 1
            value = (value << 7) | (byte & 0x7F)
            if not (byte & 0x80):
                return value, offset

    @staticmethod
    def _midi_note_to_frequency(note_number):
        return 440.0 * (2.0 ** ((float(note_number) - 69.0) / 12.0))

    @staticmethod
    def _empty_flo_dict():
        return {
            'g0': [np.array([], dtype=float), np.array([], dtype=float)],
            'g1': [np.array([], dtype=float), np.array([], dtype=float)],
            'g2': [np.array([], dtype=float), np.array([], dtype=float)],
            'rx0': [np.array([], dtype=float), np.array([], dtype=float)],
            'rx1': [np.array([], dtype=float), np.array([], dtype=float)],
            'tx0': [np.array([], dtype=float), np.array([], dtype=float)],
            'tx1': [np.array([], dtype=float), np.array([], dtype=float)],
            'ttl0': [np.array([], dtype=float), np.array([], dtype=float)],
            'ttl1': [np.array([], dtype=float), np.array([], dtype=float)],
        }

    @staticmethod
    def _initialize_flo_dict(flo_dict, t0, grad_state):
        for axis_index in range(3):
            flo_dict[f'g{axis_index}'][0] = np.array([t0], dtype=float)
            flo_dict[f'g{axis_index}'][1] = np.array([float(grad_state[axis_index])], dtype=float)
        for key in ('rx0', 'rx1', 'tx0', 'tx1', 'ttl0', 'ttl1'):
            flo_dict[key][0] = np.array([t0], dtype=float)
            flo_dict[key][1] = np.array([0.0], dtype=float)

    @staticmethod
    def _append_flo_dict_end(flo_dict, t_end, grad_state):
        for axis_index in range(3):
            flo_dict[f'g{axis_index}'][0] = np.concatenate(
                (flo_dict[f'g{axis_index}'][0], np.array([t_end], dtype=float)),
                axis=0,
            )
            flo_dict[f'g{axis_index}'][1] = np.concatenate(
                (flo_dict[f'g{axis_index}'][1], np.array([float(grad_state[axis_index])], dtype=float)),
                axis=0,
            )
        for key in ('rx0', 'rx1', 'tx0', 'tx1', 'ttl0', 'ttl1'):
            flo_dict[key][0] = np.concatenate((flo_dict[key][0], np.array([t_end], dtype=float)), axis=0)
            flo_dict[key][1] = np.concatenate((flo_dict[key][1], np.array([0.0], dtype=float)), axis=0)

    def _retime_axis_waveforms_for_hardware(self, axis_waveforms, min_event_spacing_us):
        active_axes = [
            axis_index
            for axis_index, (axis_times, _) in axis_waveforms.items()
            if axis_times.size > 0
        ]
        if active_grad_board() != 'ocra1' or len(active_axes) <= 1:
            return axis_waveforms, 0

        min_tick_spacing = max(1, int(np.ceil(min_event_spacing_us * fpga_clk_freq_MHz)))
        gradient_events = []
        for axis_index in range(3):
            axis_times, axis_values = axis_waveforms[axis_index]
            axis_ticks = np.round(np.array(axis_times, dtype=float) * fpga_clk_freq_MHz).astype(np.int64)
            for tick, value in zip(axis_ticks, axis_values):
                gradient_events.append((int(tick), axis_index, float(value)))

        gradient_events.sort(key=lambda event: (event[0], event[1]))
        retimed_events = {0: [[], []], 1: [[], []], 2: [[], []]}
        next_available_tick = None
        n_retimed_events = 0

        for tick, axis_index, value in gradient_events:
            scheduled_tick = tick
            if next_available_tick is not None and scheduled_tick < next_available_tick:
                scheduled_tick = next_available_tick
                n_retimed_events += 1

            retimed_events[axis_index][0].append(scheduled_tick / fpga_clk_freq_MHz)
            retimed_events[axis_index][1].append(value)
            next_available_tick = scheduled_tick + min_tick_spacing

        retimed_waveforms = {}
        for axis_index in range(3):
            axis_times, axis_values = retimed_events[axis_index]
            retimed_waveforms[axis_index] = (
                np.array(axis_times, dtype=float),
                np.array(axis_values, dtype=float),
            )

        return retimed_waveforms, n_retimed_events

    def _create_batch(self, batch_events, batch_start_state, batch_end_state, final_batch, min_event_spacing_us):
        all_times = [time_value for axis_events in batch_events.values() for time_value, _ in axis_events]
        if not all_times:
            return None

        lead_us = min_event_spacing_us + 1.0 / fpga_clk_freq_MHz
        batch_origin_us = min(all_times) - lead_us
        batch_end_time_us = max(all_times) - batch_origin_us + lead_us

        flo_dict = self._empty_flo_dict()
        self._initialize_flo_dict(flo_dict, 0.0, batch_start_state)

        for axis_index in range(3):
            axis_events = batch_events[axis_index]
            if not axis_events:
                continue
            axis_times = np.array([time_value - batch_origin_us for time_value, _ in axis_events], dtype=float)
            axis_values = np.array([value for _, value in axis_events], dtype=float)
            flo_dict[f'g{axis_index}'][0] = np.concatenate((flo_dict[f'g{axis_index}'][0], axis_times), axis=0)
            flo_dict[f'g{axis_index}'][1] = np.concatenate((flo_dict[f'g{axis_index}'][1], axis_values), axis=0)

        final_grad_state = np.zeros(3, dtype=float) if final_batch else np.array(batch_end_state, dtype=float)
        self._append_flo_dict_end(flo_dict, batch_end_time_us, final_grad_state)

        return {
            'flo_dict': flo_dict,
            'duration_us': batch_end_time_us,
            'n_orders': sum(len(flo_dict[key][0]) for key in flo_dict.keys()),
        }

    def _split_waveforms_into_batches(self, axis_waveforms, initial_grad_state, min_event_spacing_us):
        gradient_events = []
        for axis_index in range(3):
            axis_times, axis_values = axis_waveforms[axis_index]
            for time_value, amplitude_value in zip(axis_times, axis_values):
                gradient_events.append((float(time_value), axis_index, float(amplitude_value)))

        if not gradient_events:
            return []

        gradient_events.sort(key=lambda event: (event[0], event[1]))
        base_orders = 2 * len(self._empty_flo_dict())
        batches = []
        batch_events = {0: [], 1: [], 2: []}
        batch_gradient_event_count = 0
        current_grad_state = np.array(initial_grad_state, dtype=float)
        batch_start_state = current_grad_state.copy()
        batch_first_time = None
        max_batch_span_us = self._max_batch_span_us()
        event_index = 0

        while event_index < len(gradient_events):
            time_value = gradient_events[event_index][0]
            time_slice = []
            while event_index < len(gradient_events) and gradient_events[event_index][0] == time_value:
                time_slice.append(gradient_events[event_index])
                event_index += 1

            exceeds_order_limit = (
                batch_gradient_event_count > 0 and
                base_orders + batch_gradient_event_count + len(time_slice) > hw.maxOrders
            )
            exceeds_time_limit = (
                batch_gradient_event_count > 0 and
                batch_first_time is not None and
                time_value - batch_first_time > max_batch_span_us
            )
            if exceeds_order_limit or exceeds_time_limit:
                batch = self._create_batch(
                    batch_events=batch_events,
                    batch_start_state=batch_start_state,
                    batch_end_state=current_grad_state,
                    final_batch=False,
                    min_event_spacing_us=min_event_spacing_us,
                )
                if batch is not None:
                    batches.append(batch)
                batch_events = {0: [], 1: [], 2: []}
                batch_gradient_event_count = 0
                batch_start_state = current_grad_state.copy()
                batch_first_time = None

            if batch_first_time is None:
                batch_first_time = time_value

            for _, axis_index, amplitude_value in time_slice:
                batch_events[axis_index].append((time_value, amplitude_value))
                current_grad_state[axis_index] = amplitude_value
            batch_gradient_event_count += len(time_slice)

        batch = self._create_batch(
            batch_events=batch_events,
            batch_start_state=batch_start_state,
            batch_end_state=current_grad_state,
            final_batch=True,
            min_event_spacing_us=min_event_spacing_us,
        )
        if batch is not None:
            batches.append(batch)

        for batch in batches:
            if batch['n_orders'] > hw.maxOrders:
                raise ValueError(
                    f"Automatic MIDI batching failed: one batch still needs {batch['n_orders']} "
                    f"events, above the hardware limit of {hw.maxOrders}."
                )

        return batches

    @staticmethod
    def _sequence_directory():
        return os.path.dirname(os.path.abspath(__file__))

    def _bundled_midi_files(self):
        midi_files = []
        for root, dirs, files in os.walk(self._sequence_directory()):
            dirs[:] = [dirname for dirname in dirs if dirname != '__pycache__']
            for filename in files:
                if filename.lower().endswith(('.mid', '.midi')):
                    midi_files.append(os.path.join(root, filename))
        midi_files.sort()
        return midi_files

    def _resolve_midi_path(self, midi_file):
        midi_file = str(midi_file).strip()
        if not midi_file:
            bundled_midi_files = self._bundled_midi_files()
            if bundled_midi_files:
                return bundled_midi_files[0]
            raise ValueError(
                f"No MIDI file was provided and no bundled .mid/.midi files were found in "
                f"{self._sequence_directory()}."
            )

        expanded_path = os.path.expanduser(midi_file)
        candidate_paths = []
        if os.path.isabs(expanded_path):
            candidate_paths.append(expanded_path)
        else:
            candidate_paths.extend([
                os.path.abspath(expanded_path),
                os.path.join(self._sequence_directory(), expanded_path),
                os.path.join(self._sequence_directory(), 'examples', expanded_path),
            ])

        checked_paths = []
        for candidate_path in candidate_paths:
            normalized_path = os.path.abspath(candidate_path)
            if normalized_path in checked_paths:
                continue
            checked_paths.append(normalized_path)
            if os.path.isfile(normalized_path):
                return normalized_path

        raise FileNotFoundError(f"MIDI file not found. Checked: {', '.join(checked_paths)}")

    def _get_midi_intervals(self, midi_file, include_percussion):
        midi_path = self._resolve_midi_path(midi_file)
        cache_key = (midi_path, os.path.getmtime(midi_path), int(include_percussion))
        if self._cached_midi_key == cache_key and self._cached_intervals is not None:
            return midi_path, [
                _MidiNoteInterval(
                    start_tick=interval.start_tick,
                    end_tick=interval.end_tick,
                    note=interval.note,
                    velocity=interval.velocity,
                    track=interval.track,
                    channel=interval.channel,
                    start_s=interval.start_s,
                    end_s=interval.end_s,
                )
                for interval in self._cached_intervals
            ]

        intervals = self._parse_midi_file(midi_path, bool(include_percussion))
        self._cached_midi_key = cache_key
        self._cached_intervals = intervals
        return midi_path, [
            _MidiNoteInterval(
                start_tick=interval.start_tick,
                end_tick=interval.end_tick,
                note=interval.note,
                velocity=interval.velocity,
                track=interval.track,
                channel=interval.channel,
                start_s=interval.start_s,
                end_s=interval.end_s,
            )
            for interval in intervals
        ]

    def _parse_midi_file(self, midi_path, include_percussion):
        with open(midi_path, 'rb') as midi_stream:
            raw = midi_stream.read()

        if len(raw) < 14 or raw[:4] != b'MThd':
            raise ValueError("Invalid MIDI header.")

        header_length = struct.unpack('>I', raw[4:8])[0]
        if header_length != 6:
            raise ValueError("Unsupported MIDI header length.")

        midi_format, track_count, division = struct.unpack('>HHH', raw[8:14])
        if midi_format not in (0, 1):
            raise ValueError(f"Unsupported MIDI format {midi_format}. Only format 0 and 1 are supported.")
        if division & 0x8000:
            raise ValueError("SMPTE-based MIDI timing is not supported.")

        ticks_per_quarter = division
        tempo_events = [(0, self._DEFAULT_TEMPO_US_PER_QUARTER)]
        note_intervals = []

        offset = 8 + header_length
        for track_index in range(track_count):
            if raw[offset:offset + 4] != b'MTrk':
                raise ValueError(f"Invalid MIDI track header in track {track_index}.")
            track_length = struct.unpack('>I', raw[offset + 4:offset + 8])[0]
            track_data = memoryview(raw[offset + 8:offset + 8 + track_length])
            offset += 8 + track_length

            absolute_tick = 0
            cursor = 0
            running_status = None
            active_notes = {}

            while cursor < len(track_data):
                delta, cursor = self._read_vlq(track_data, cursor)
                absolute_tick += delta
                if cursor >= len(track_data):
                    break

                status = track_data[cursor]
                if status < 0x80:
                    if running_status is None:
                        raise ValueError("Running-status MIDI message without previous status.")
                    status = running_status
                else:
                    cursor += 1
                    if status < 0xF0:
                        running_status = status

                if status == 0xFF:
                    if cursor >= len(track_data):
                        raise ValueError("Malformed MIDI meta event.")
                    meta_type = track_data[cursor]
                    cursor += 1
                    meta_length, cursor = self._read_vlq(track_data, cursor)
                    meta_data = bytes(track_data[cursor:cursor + meta_length])
                    cursor += meta_length
                    if meta_type == 0x51 and len(meta_data) == 3:
                        tempo_events.append((absolute_tick, int.from_bytes(meta_data, byteorder='big')))
                    if meta_type == 0x2F:
                        break
                    continue

                if status in (0xF0, 0xF7):
                    data_length, cursor = self._read_vlq(track_data, cursor)
                    cursor += data_length
                    continue

                message_type = status & 0xF0
                channel = status & 0x0F

                if message_type in (0xC0, 0xD0):
                    cursor += 1
                    continue

                if cursor + 2 > len(track_data):
                    raise ValueError("Malformed MIDI channel message.")
                note_number = int(track_data[cursor])
                velocity = int(track_data[cursor + 1])
                cursor += 2

                if channel == 9 and not include_percussion:
                    continue

                if message_type == 0x90 and velocity > 0:
                    active_notes.setdefault((channel, note_number), []).append((absolute_tick, velocity))
                elif message_type == 0x80 or (message_type == 0x90 and velocity == 0):
                    key = (channel, note_number)
                    if key in active_notes and active_notes[key]:
                        start_tick, start_velocity = active_notes[key].pop()
                        if absolute_tick > start_tick:
                            note_intervals.append(
                                _MidiNoteInterval(
                                    start_tick=start_tick,
                                    end_tick=absolute_tick,
                                    note=note_number,
                                    velocity=start_velocity,
                                    track=track_index,
                                    channel=channel,
                                )
                            )

            for (channel, note_number), pending_notes in active_notes.items():
                while pending_notes:
                    start_tick, start_velocity = pending_notes.pop()
                    if absolute_tick > start_tick:
                        note_intervals.append(
                            _MidiNoteInterval(
                                start_tick=start_tick,
                                end_tick=absolute_tick,
                                note=note_number,
                                velocity=start_velocity,
                                track=track_index,
                                channel=channel,
                            )
                        )

        tempo_ticks, tempo_values, tempo_accumulated = self._build_tempo_map(tempo_events, ticks_per_quarter)
        for interval in note_intervals:
            interval.start_s = self._tick_to_seconds(interval.start_tick, ticks_per_quarter, tempo_ticks, tempo_values, tempo_accumulated)
            interval.end_s = self._tick_to_seconds(interval.end_tick, ticks_per_quarter, tempo_ticks, tempo_values, tempo_accumulated)

        note_intervals.sort(key=lambda interval: (interval.start_s, interval.end_s, interval.note))
        return note_intervals

    def _build_tempo_map(self, tempo_events, ticks_per_quarter):
        tempo_events.sort(key=lambda item: item[0])
        merged_events = []
        for tick, tempo in tempo_events:
            if merged_events and merged_events[-1][0] == tick:
                merged_events[-1] = (tick, tempo)
            else:
                merged_events.append((tick, tempo))

        tempo_ticks = np.array([item[0] for item in merged_events], dtype=np.int64)
        tempo_values = np.array([item[1] for item in merged_events], dtype=np.float64)
        tempo_accumulated = np.zeros(len(merged_events), dtype=np.float64)

        for index in range(1, len(merged_events)):
            tick_delta = tempo_ticks[index] - tempo_ticks[index - 1]
            tempo_accumulated[index] = tempo_accumulated[index - 1] + (
                tick_delta * tempo_values[index - 1] / 1e6 / ticks_per_quarter
            )

        return tempo_ticks, tempo_values, tempo_accumulated

    @staticmethod
    def _tick_to_seconds(tick, ticks_per_quarter, tempo_ticks, tempo_values, tempo_accumulated):
        segment = bisect_right(tempo_ticks, tick) - 1
        segment = max(segment, 0)
        return tempo_accumulated[segment] + (
            (tick - tempo_ticks[segment]) * tempo_values[segment] / 1e6 / ticks_per_quarter
        )

    def _partition_voices(self, intervals):
        voices = []
        voice_end_times = []

        for interval in intervals:
            selected_index = None
            selected_end = -np.inf
            for index, end_time in enumerate(voice_end_times):
                if end_time <= interval.start_s and end_time > selected_end:
                    selected_index = index
                    selected_end = end_time

            if selected_index is None:
                voices.append([interval])
                voice_end_times.append(interval.end_s)
            else:
                voices[selected_index].append(interval)
                voice_end_times[selected_index] = interval.end_s

        def voice_priority(voice):
            durations = [max(interval.end_s - interval.start_s, 0.0) for interval in voice]
            total_duration = float(np.sum(durations))
            if total_duration <= 0:
                return (0.0, 0.0, 0.0)
            weighted_pitch = float(np.sum([interval.note * duration for interval, duration in zip(voice, durations)]))
            weighted_velocity = float(np.sum([interval.velocity * duration for interval, duration in zip(voice, durations)]))
            return (
                weighted_pitch / total_duration,
                total_duration,
                weighted_velocity / total_duration,
            )

        voices.sort(key=voice_priority, reverse=True)
        return voices

    def _select_voices(self, voices, axis_name):
        axis_names = {0: 'x', 1: 'y', 2: 'z'}
        if axis_name in ('x', 'y', 'z'):
            single_axis_index = {'x': 0, 'y': 1, 'z': 2}[axis_name]
            voice_index = max(int(self.singleVoice) - 1, 0)
            if voice_index >= len(voices):
                print(
                    f"WARNING: Requested voice {voice_index + 1} does not exist. "
                    f"Using voice {len(voices)} instead."
                )
                voice_index = len(voices) - 1
            return [{
                'axis_index': single_axis_index,
                'axis_name': axis_names[single_axis_index],
                'voice_index': voice_index,
                'intervals': voices[voice_index],
            }]

        selected = []
        for axis_index, intervals in zip((0, 1, 2), voices[:3]):
            selected.append({
                'axis_index': axis_index,
                'axis_name': axis_names[axis_index],
                'voice_index': len(selected),
                'intervals': intervals,
            })
        return selected

    def _build_voice_waveform(self, intervals, amplitude, transpose, max_frequency, playback_start_us, shimming, min_event_spacing_us):
        event_times = []
        event_values = []

        for interval in intervals:
            start_us = playback_start_us + interval.start_s * 1e6
            end_us = playback_start_us + interval.end_s * 1e6
            if end_us <= start_us:
                continue

            frequency_hz = self._midi_note_to_frequency(interval.note + transpose)
            frequency_hz = min(max_frequency, frequency_hz)

            note_times, note_values = self._build_square_wave(start_us, end_us, amplitude, frequency_hz)
            for time_us, value in zip(note_times, note_values):
                if event_times and time_us <= event_times[-1]:
                    if time_us == event_times[-1]:
                        event_values[-1] = value
                    continue
                event_times.append(time_us)
                event_values.append(value)

        if not event_times:
            return np.array([], dtype=float), np.array([], dtype=float)

        min_tick_spacing = max(1, int(np.ceil(min_event_spacing_us * fpga_clk_freq_MHz)))
        raw_ticks = np.round(np.array(event_times, dtype=float) * fpga_clk_freq_MHz).astype(np.int64)
        final_ticks = []
        final_values = []

        for tick, value in zip(raw_ticks, event_values):
            if final_ticks and tick == final_ticks[-1]:
                final_values[-1] = value + shimming
                continue
            if final_ticks and tick - final_ticks[-1] < min_tick_spacing:
                tick = final_ticks[-1] + min_tick_spacing
            final_ticks.append(tick)
            final_values.append(value + shimming)

        return np.array(final_ticks, dtype=float) / fpga_clk_freq_MHz, np.array(final_values, dtype=float)

    def _build_square_wave(self, start_us, end_us, amplitude, frequency_hz):
        duration_us = end_us - start_us
        if duration_us <= 0:
            return [], []

        half_period_us = 5e5 / float(frequency_hz)
        clock_period_us = 1.0 / fpga_clk_freq_MHz

        times = [float(start_us)]
        values = [float(amplitude)]
        current_value = float(amplitude)
        edge_time = start_us + half_period_us

        while edge_time < end_us - 0.5 * clock_period_us:
            current_value *= -1.0
            times.append(float(edge_time))
            values.append(float(current_value))
            edge_time += half_period_us

        times.append(float(end_us))
        values.append(0.0)
        return times, values

    def _voice_plot_data(self, intervals, transpose, max_frequency):
        times_ms = []
        freq_hz = []
        last_end_ms = 0.0

        for interval in intervals:
            start_ms = interval.start_s * 1e3
            end_ms = interval.end_s * 1e3
            frequency = min(max_frequency, self._midi_note_to_frequency(interval.note + transpose))
            if start_ms > last_end_ms:
                times_ms.extend([last_end_ms, start_ms])
                freq_hz.extend([0.0, 0.0])
            times_ms.extend([start_ms, start_ms, end_ms, end_ms])
            freq_hz.extend([0.0, frequency, frequency, 0.0])
            last_end_ms = end_ms

        if not times_ms:
            return [], []

        return times_ms, freq_hz
