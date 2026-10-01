"""Offline checks for transcript integrity, resume behavior, and input safety."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import diarize
import engine
from languages import ROMANIAN_ALIGNMENT_MODEL
from transcript import render_transcript, safe_title, speaker_turns, timestamp


class InputTests(unittest.TestCase):
    def test_romanian_aliases_share_inference_settings(self):
        expected = diarize.inference_settings(diarize.parser().parse_args(['--language', 'ro']))
        for value in ['RO', 'Romanian', 'romana', 'română', 'roma\u0302na\u0306', 'ron', 'rum', 'ro-RO', 'ro_MD']:
            with self.subTest(language=value):
                actual = diarize.parser().parse_args(['--language', value])
                self.assertEqual(diarize.inference_settings(actual), expected)
        self.assertIsNone(diarize.parser().parse_args(['--language', 'auto']).language)
        self.assertIsNone(diarize.parser().parse_args([]).language)

    def test_romanian_rejects_english_only_model_before_download(self):
        with redirect_stderr(io.StringIO()) as errors, patch('diarize.preflight') as preflight:
            with self.assertRaises(SystemExit) as result:
                diarize.main(['--language', 'Romanian', '--model', 'base.en', '--dry-run'])
        self.assertEqual(result.exception.code, 2)
        self.assertIn('without .en', errors.getvalue())
        preflight.assert_not_called()

    def test_credentials_support_existing_file_key_and_environment_precedence(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / 'env.txt'
            path.write_text('HUGGING_FACE_TOKEN="hf_example123"\n', encoding='utf-8')
            diarize.load_credentials(path)
            self.assertEqual(os.environ['HF_TOKEN'], 'hf_example123')
            os.environ['HF_TOKEN'] = 'hf_environment456'
            diarize.load_credentials(path)
            self.assertEqual(os.environ['HF_TOKEN'], 'hf_environment456')

    def test_youtube_aliases_are_one_job(self):
        urls = ['https://youtu.be/3V33MIIcXjM?is=tracking', 'https://www.youtube.com/live/3V33MIIcXjM?x=1', 'https://www.youtube.com/watch?v=3V33MIIcXjM&list=abc']
        self.assertEqual(len(diarize.read_jobs(None, urls)), 1)

    def test_bom_blanks_and_comments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'jobs.txt'
            path.write_text('\ufeff# header\n\nhttps://youtu.be/3V33MIIcXjM\n', encoding='utf-8')
            self.assertEqual(len(diarize.read_jobs(path)), 1)

    def test_both_supplied_rumble_urls(self):
        urls = [
            'https://rumble.com/v4cw3gr-the-king-of-toxic-masculinity-a-conversation-with-cobra-tate-in-warsaw-pola.html',
            'https://rumble.com/v4cw82u-m2-discuss-their-recent-excursion-to-eastern-europe-and-visiting-their-frie.html',
        ]
        self.assertEqual(diarize.read_jobs(None, urls), urls)

    def test_reject_channels_playlists_and_lookalike_domains(self):
        for url in ['https://youtube.com/playlist?list=abc', 'https://rumble.com/c/channel', 'https://youtube.com.evil.example/watch?v=3V33MIIcXjM', 'file:///video.mp4', 'https://youtu.be/too-short']:
            with self.subTest(url=url), self.assertRaises(ValueError):
                diarize.canonical_url(url)


class TranscriptTests(unittest.TestCase):
    def test_romanian_diacritics_and_speaker_changes_survive_utf8_output(self):
        source = {'language': 'ro', 'segments': [{'start': 0, 'end': 3, 'words': [
            {'word': 'Bună,', 'start': 0, 'end': 0.5, 'speaker': 'A'},
            {'word': 'țară!', 'start': 0.5, 'end': 1, 'speaker': 'A'},
            {'word': 'Și', 'start': 1, 'end': 1.5, 'speaker': 'B'},
            {'word': 'mâine', 'start': 1.5, 'end': 2, 'speaker': 'B'},
            {'word': 'învățăm.', 'start': 2, 'end': 3, 'speaker': 'B'},
        ]}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'romanian.txt'
            diarize.atomic_text(path, render_transcript({'title': 'Discuție în România'}, source, 'base', 'url'))
            text = path.read_text(encoding='utf-8')
        self.assertIn('Language: ro', text)
        self.assertTrue(text.startswith('Discuție în România\n'))
        self.assertIn('[00:00:00.000 --> 00:00:01.000] SPEAKER 01: Bună, țară!', text)
        self.assertIn('[00:00:01.000 --> 00:00:03.000] SPEAKER 02: Și mâine învățăm.', text)

    def test_speaker_change_inside_one_sentence(self):
        segments = [{'start': 0, 'end': 4, 'speaker': 'A', 'text': 'Hello. Yes!', 'words': [{'word': 'Hello.', 'start': 0, 'end': 1, 'speaker': 'A'}, {'word': 'Yes!', 'start': 2, 'end': 3, 'speaker': 'B'}]}]
        turns = speaker_turns(segments)
        self.assertEqual([turn['speaker'] for turn in turns], ['A', 'B'])
        self.assertEqual([turn['text'] for turn in turns], ['Hello.', 'Yes!'])

    def test_unaligned_numbers_and_missing_words_survive(self):
        turns = speaker_turns([{'start': 10, 'end': 12, 'speaker': 'A', 'text': 'In 2014.', 'words': [{'word': 'In', 'start': 10, 'end': 11}, {'word': '2014.'}]}])
        self.assertEqual(turns[0]['text'], 'In 2014.')
        self.assertEqual(speaker_turns([{'start': 2, 'end': 5, 'text': 'Unaligned but retained.'}])[0]['text'], 'Unaligned but retained.')

    def test_all_seven_speakers_get_distinct_labels(self):
        segments = [{'start': i, 'end': i + 0.5, 'speaker': f'original_{i}', 'text': f'Turn {i}.'} for i in range(7)]
        text = render_transcript({'title': 'Actual video title'}, {'language': 'en', 'segments': segments}, 'base', 'https://rumble.com/v123.html')
        self.assertTrue(text.startswith('Actual video title\n'))
        self.assertIn('Speakers detected: 7', text)
        self.assertIn('SPEAKER 07:', text)
        self.assertIn('[00:00:06.000 --> 00:00:06.500]', text)

    def test_unknown_is_visible_and_preview_explicit(self):
        text = render_transcript({'title': 'Title'}, {'segments': [{'start': 0, 'end': 1, 'text': 'Hello'}]}, 'base', 'url', 60)
        self.assertIn('SPEAKER UNKNOWN:', text)
        self.assertIn('PREVIEW ONLY: first 60 seconds', text)

    def test_timestamps_and_windows_names(self):
        self.assertEqual(timestamp(3599.9999), '01:00:00.000')
        self.assertEqual(timestamp(36001.123), '10:00:01.123')
        self.assertEqual(safe_title('CON'), '_CON')
        self.assertNotIn(':', safe_title('Interview: A/B?'))
        self.assertLessEqual(len(safe_title('Long ' * 100)), 100)


class LanguagePipelineTests(unittest.TestCase):
    def test_explicit_and_detected_romanian_alignment_across_chunks(self):
        # Exercise worker entry points without downloading/loading ML models.
        for requested, detected in [('ro', 'ro'), (None, 'ro'), ('en', 'en')]:
            with self.subTest(requested=requested, detected=detected), tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                config_path = work / 'config.json'
                config_path.write_text(json.dumps({
                    'threads': 1, 'model_dir': str(work / 'models'), 'audio': 'mock.f32',
                    'device': 'cpu', 'chunk_minutes': 1, 'model': 'base',
                    'compute_type': 'int8', 'language': requested, 'batch_size': 1,
                }), encoding='utf-8')
                sentence = 'Bună, România!' if detected == 'ro' else 'Hello!'
                model = Mock()
                model.vad_model.return_value = [SimpleNamespace(start=0, end=1)]
                model.detect_language.return_value = detected
                model.transcribe.side_effect = lambda *args, **kwargs: {
                    'language': detected, 'segments': [{'start': 0, 'end': 1, 'text': sentence}],
                }
                whisperx = Mock()
                whisperx.load_model.return_value = model
                whisperx.load_align_model.return_value = (object(), {'language': detected})
                whisperx.align.side_effect = lambda segments, *args, **kwargs: {'segments': [
                    dict(s, words=[{'word': s['text'], 'start': s['start'], 'end': s['end']}]) for s in segments
                ]}
                numpy = SimpleNamespace(float32='float32', memmap=Mock(return_value=range(2 * 60 * engine.SAMPLE_RATE)))
                alignment = SimpleNamespace(DEFAULT_ALIGN_MODELS_HF={'ro': ROMANIAN_ALIGNMENT_MODEL}, DEFAULT_ALIGN_MODELS_TORCH={'en': 'english-model'})
                with patch.dict(sys.modules, {'numpy': numpy, 'torch': Mock(), 'whisperx': whisperx, 'whisperx.alignment': alignment}), patch.dict(os.environ), redirect_stdout(io.StringIO()):
                    with patch.object(sys, 'argv', ['engine.py', str(config_path), 'transcribe']):
                        self.assertEqual(engine.main(), 0)
                    with patch.object(sys, 'argv', ['engine.py', str(config_path), 'align']):
                        self.assertEqual(engine.main(), 0)
                self.assertEqual(whisperx.load_model.call_args.kwargs['task'], 'transcribe')
                self.assertEqual([c.kwargs['language'] for c in model.transcribe.call_args_list], [detected, detected])
                self.assertEqual(model.detect_language.call_count, 2 if requested is None else 0)
                self.assertTrue(all(c.kwargs['task'] == 'transcribe' for c in model.transcribe.call_args_list))
                options = whisperx.load_align_model.call_args.kwargs
                self.assertEqual(options['language_code'], detected)
                self.assertEqual(options['model_name'], ROMANIAN_ALIGNMENT_MODEL if detected == 'ro' else None)
                result = json.loads((work / 'aligned.json').read_text(encoding='utf-8'))
                self.assertEqual(result['language'], detected)
                self.assertEqual([s['text'] for s in result['segments']], [sentence, sentence])
                self.assertEqual([s['words'][0]['start'] for s in result['segments']], [0, 60])
                self.assertEqual([s['words'][0]['end'] for s in result['segments']], [1, 61])

    def test_default_command_switches_languages_within_video_and_aligns_in_order(self):
        args = diarize.parser().parse_args(['--workers', '3'])
        self.assertIsNone(args.language)
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            config_path = work / 'config.json'
            config = {**diarize.inference_settings(args), 'threads': 1, 'model_dir': str(work / 'models'),
                      'audio': 'mock.f32', 'chunk_minutes': 1, 'compute_type': 'int8'}
            config_path.write_text(json.dumps(config), encoding='utf-8')
            model = Mock()
            model.vad_model.return_value = [SimpleNamespace(start=s, end=s + 4) for s in (1, 8, 32, 40)]
            model.detect_language.side_effect = ['en', 'ro', 'en', 'ro']
            model.transcribe.side_effect = lambda *a, **kw: {'language': kw['language'], 'segments': [
                {'start': 0, 'end': 2, 'text': 'Bună, România!' if kw['language'] == 'ro' else 'Hello!'}]}
            whisperx = Mock()
            whisperx.load_model.return_value = model
            whisperx.load_align_model.side_effect = lambda **kw: (object(), {'language': kw['language_code']})
            aligned_inputs = []

            def align(segments, model, metadata, audio, device, **kwargs):
                aligned_inputs.append((metadata['language'], [s['text'] for s in segments]))
                # Real WhisperX reconstructs segments, so it drops custom language fields.
                return {'segments': [{'start': s['start'], 'end': s['end'], 'text': s['text'],
                                      'words': [{'word': s['text'], 'start': s['start'], 'end': s['end']}]}
                                     for s in segments]}

            whisperx.align.side_effect = align
            numpy = SimpleNamespace(float32='float32', memmap=Mock(return_value=range(120 * engine.SAMPLE_RATE)))
            alignment = SimpleNamespace(DEFAULT_ALIGN_MODELS_HF={'ro': ROMANIAN_ALIGNMENT_MODEL}, DEFAULT_ALIGN_MODELS_TORCH={'en': 'english-model'})
            with patch.dict(sys.modules, {'numpy': numpy, 'torch': Mock(), 'whisperx': whisperx, 'whisperx.alignment': alignment}), patch.dict(os.environ), redirect_stdout(io.StringIO()):
                for stage in ('transcribe', 'align'):
                    with patch.object(sys, 'argv', ['engine.py', str(config_path), stage]):
                        self.assertEqual(engine.main(), 0)
            self.assertEqual(model.detect_language.call_count, 4)
            self.assertEqual(whisperx.load_model.call_args.kwargs['vad_options']['chunk_size'], 30)
            self.assertEqual([c.kwargs['language_code'] for c in whisperx.load_align_model.call_args_list], ['en', 'ro'])
            self.assertEqual(whisperx.load_align_model.call_args.kwargs['model_name'], ROMANIAN_ALIGNMENT_MODEL)
            for language, texts in aligned_inputs:
                self.assertEqual(set(texts), {'Hello!'} if language == 'en' else {'Bună, România!'})
            result = json.loads((work / 'aligned.json').read_text(encoding='utf-8'))
            self.assertEqual(result['language'], 'mixed')
            self.assertEqual(result['languages'], ['en', 'ro'])
            self.assertEqual([s['language'] for s in result['segments']], ['en', 'ro', 'en', 'ro'])
            self.assertEqual([s['words'][0]['start'] for s in result['segments']], [1, 32, 61, 92])
            rendered = render_transcript({'title': 'Mixed discussion'}, result, 'base', 'url')
            self.assertIn('Languages detected: en, ro', rendered)
            self.assertEqual(rendered.count('Bună, România!'), 2)
            self.assertEqual(rendered.count('Hello!'), 2)

    def test_many_short_fragments_use_bounded_context_and_keep_offsets(self):
        model = Mock()
        model.vad_model.return_value = [SimpleNamespace(start=s, end=s + 1) for s in range(0, 60, 2)]
        model.detect_language.side_effect = ['en', 'ro']
        model.transcribe.return_value = {'segments': [{'start': 0, 'end': 1, 'text': 'speech'}]}
        result = engine.transcribe_block(model, range(60 * engine.SAMPLE_RATE), None, 1)
        self.assertEqual(model.detect_language.call_count, 2)
        self.assertEqual(model.transcribe.call_count, 2)
        self.assertEqual([s['start'] for s in result], [0, 30])
        self.assertEqual([s['language'] for s in result], ['en', 'ro'])

    def test_transcription_crash_resumes_last_completed_minute(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            config = {'threads': 1, 'model_dir': str(work), 'audio': 'mock.f32', 'device': 'cpu',
                      'chunk_minutes': 10, 'model': 'base', 'compute_type': 'int8', 'language': 'en', 'batch_size': 1}
            engine.save(work / 'config.json', config)
            numpy = SimpleNamespace(float32='float32', memmap=Mock(return_value=range(120 * engine.SAMPLE_RATE)))
            whisperx = Mock()
            model = whisperx.load_model.return_value
            speech = {'segments': [{'start': 1, 'end': 2, 'text': 'saved'}]}
            model.transcribe.side_effect = [speech, RuntimeError('interrupted')]
            with patch.dict(sys.modules, {'numpy': numpy, 'torch': Mock(), 'whisperx': whisperx}), patch.dict(os.environ), redirect_stdout(io.StringIO()), patch.object(sys, 'argv', ['engine.py', str(work / 'config.json'), 'transcribe']):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    engine.main()
                self.assertFalse((work / 'transcribed.json').exists())
                model.transcribe.reset_mock(side_effect=True)
                model.transcribe.return_value = speech
                self.assertEqual(engine.main(), 0)
            self.assertEqual(model.transcribe.call_count, 1)
            result = json.loads((work / 'transcribed.json').read_text(encoding='utf-8'))
            self.assertEqual([s['start'] for s in result['segments']], [1, 61])

    def test_alignment_download_failure_retains_text_and_allows_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            engine.save(work / 'config.json', {'threads': 1, 'model_dir': str(work), 'audio': 'mock.f32', 'device': 'cpu', 'chunk_minutes': 1})
            source = {'language': 'mixed', 'languages': ['en', 'zh'], 'segments': [
                {'start': 1, 'end': 2, 'text': 'Hello', 'language': 'en'},
                {'start': 3, 'end': 4, 'text': '保留', 'language': 'zh'}]}
            engine.save(work / 'transcribed.json', source)
            whisperx = Mock()
            whisperx.load_align_model.side_effect = [(object(), {}), ValueError('download failed')]
            whisperx.align.side_effect = lambda segments, *a, **kw: {'segments': segments}
            numpy = SimpleNamespace(float32='float32', memmap=Mock(return_value=range(60 * engine.SAMPLE_RATE)))
            alignment = SimpleNamespace(DEFAULT_ALIGN_MODELS_HF={'zh': 'chinese'}, DEFAULT_ALIGN_MODELS_TORCH={'en': 'english'})
            with patch.dict(sys.modules, {'numpy': numpy, 'torch': Mock(), 'whisperx': whisperx, 'whisperx.alignment': alignment}), patch.dict(os.environ), redirect_stdout(io.StringIO()), patch.object(sys, 'argv', ['engine.py', str(work / 'config.json'), 'align']):
                self.assertEqual(engine.main(), 0)
            result = json.loads((work / 'aligned.json').read_text(encoding='utf-8'))
            self.assertEqual(result['segments'], source['segments'])
            self.assertIn('zh', result['alignment_warnings'][0])
            self.assertIn('Timing note:', render_transcript({'title': 'test'}, result, 'base', 'url'))

    def test_alignment_crash_reuses_completed_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            engine.save(work / 'config.json', {'threads': 1, 'model_dir': str(work), 'audio': 'mock.f32', 'device': 'cpu', 'chunk_minutes': 1})
            engine.save(work / 'transcribed.json', {'language': 'en', 'segments': [
                {'start': 1, 'end': 2, 'text': 'first'}, {'start': 61, 'end': 62, 'text': 'second'}]})
            whisperx = Mock()
            whisperx.load_align_model.return_value = (object(), {})
            whisperx.align.side_effect = [{'segments': [{'start': 1, 'end': 2, 'text': 'first', 'words': [{'word': 'first', 'start': 1, 'end': 2}]}]}, RuntimeError('interrupted')]
            numpy = SimpleNamespace(float32='float32', memmap=Mock(return_value=range(120 * engine.SAMPLE_RATE)))
            alignment = SimpleNamespace(DEFAULT_ALIGN_MODELS_HF={}, DEFAULT_ALIGN_MODELS_TORCH={'en': 'english'})
            with patch.dict(sys.modules, {'numpy': numpy, 'torch': Mock(), 'whisperx': whisperx, 'whisperx.alignment': alignment}), patch.dict(os.environ), redirect_stdout(io.StringIO()), patch.object(sys, 'argv', ['engine.py', str(work / 'config.json'), 'align']):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    engine.main()
                whisperx.align.reset_mock(side_effect=True)
                whisperx.align.return_value = {'segments': [{'start': 1, 'end': 2, 'text': 'second'}]}
                self.assertEqual(engine.main(), 0)
            self.assertEqual(whisperx.align.call_count, 1)
            result = json.loads((work / 'aligned.json').read_text(encoding='utf-8'))
            self.assertEqual([s['start'] for s in result['segments']], [1, 61])

    def test_isolated_language_detection_does_not_download_another_aligner(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            engine.save(work / 'config.json', {'threads': 1, 'model_dir': str(work), 'audio': 'mock.f32', 'device': 'cpu', 'chunk_minutes': 10})
            source = {'language': 'mixed', 'segments': [
                {'start': 1, 'end': 21, 'text': 'English speech', 'language': 'en'},
                {'start': 40, 'end': 40.5, 'text': '保留', 'language': 'zh'}]}
            engine.save(work / 'transcribed.json', source)
            whisperx = Mock()
            whisperx.load_align_model.return_value = (object(), {})
            whisperx.align.side_effect = lambda segments, *a, **kw: {'segments': segments}
            numpy = SimpleNamespace(float32='float32', memmap=Mock(return_value=range(600 * engine.SAMPLE_RATE)))
            alignment = SimpleNamespace(DEFAULT_ALIGN_MODELS_HF={'zh': 'chinese'}, DEFAULT_ALIGN_MODELS_TORCH={'en': 'english'})
            with patch.dict(sys.modules, {'numpy': numpy, 'torch': Mock(), 'whisperx': whisperx, 'whisperx.alignment': alignment}), patch.dict(os.environ), redirect_stdout(io.StringIO()), patch.object(sys, 'argv', ['engine.py', str(work / 'config.json'), 'align']):
                self.assertEqual(engine.main(), 0)
            self.assertEqual([c.kwargs['language_code'] for c in whisperx.load_align_model.call_args_list], ['en'])
            result = json.loads((work / 'aligned.json').read_text(encoding='utf-8'))
            self.assertEqual(result['segments'], source['segments'])
            self.assertIn('Brief detected language zh', result['alignment_warnings'][0])

    def test_silence_does_not_trigger_language_detection_or_transcription(self):
        model = Mock()
        model.vad_model.return_value = []
        self.assertEqual(engine.transcribe_block(model, range(16000), None, 1), [])
        model.detect_language.assert_not_called()
        model.transcribe.assert_not_called()

    def test_unsupported_alignment_language_retains_text_and_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            config_path = work / 'config.json'
            config_path.write_text(json.dumps({'threads': 1, 'model_dir': str(work), 'audio': 'mock.f32',
                                               'device': 'cpu', 'chunk_minutes': 1}), encoding='utf-8')
            source = {'language': 'xx', 'languages': ['xx'], 'segments': [{'start': 1, 'end': 2, 'text': 'Retained speech', 'language': 'xx'}]}
            engine.save(work / 'transcribed.json', source)
            whisperx = Mock()
            numpy = SimpleNamespace(float32='float32', memmap=Mock(return_value=range(160000)))
            alignment = SimpleNamespace(DEFAULT_ALIGN_MODELS_HF={}, DEFAULT_ALIGN_MODELS_TORCH={})
            with patch.dict(sys.modules, {'numpy': numpy, 'torch': Mock(), 'whisperx': whisperx, 'whisperx.alignment': alignment}), patch.dict(os.environ), redirect_stdout(io.StringIO()), patch.object(sys, 'argv', ['engine.py', str(config_path), 'align']):
                self.assertEqual(engine.main(), 0)
            whisperx.load_align_model.assert_not_called()
            self.assertEqual(json.loads((work / 'aligned.json').read_text(encoding='utf-8')), source)


class JobTests(unittest.TestCase):
    def setUp(self):
        diarize.STOP.clear()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.args = argparse.Namespace(cache_dir=self.root / 'cache', out=self.root / 'out', log_dir=self.root / 'logs', model_dir=self.root / 'models', model='base', device='cpu', compute_type='int8', language='en', batch_size=1, speakers=None, min_speakers=None, max_speakers=None, chunk_minutes=10, preview_seconds=None, threads=1, force=False, download_only=False, keep_audio=False)
        self.url = 'https://www.youtube.com/watch?v=3V33MIIcXjM'

    def tearDown(self):
        diarize.STOP.clear()
        self.temp.cleanup()

    def download(self, url, work, args, job_id):
        (work / 'audio.f32').write_bytes(b'\0' * 16)
        return {'title': 'Real title', 'url': url}

    def stage(self, command, log, label, **kwargs):
        work = Path(command[2]).parent
        checkpoint = {'transcribe': 'transcribed.json', 'align': 'aligned.json', 'diarize': 'diarized.json'}[command[3]]
        diarize.atomic_json(work / checkpoint, {'language': 'en', 'segments': [{'start': 0, 'end': 1, 'speaker': 'S', 'text': 'hello'}]})

    def test_main_command_recomputes_legacy_pipeline_then_resumes_new_results(self):
        argv = ['--workers', '3', '--url', self.url,
                '--out', str(self.args.out), '--cache-dir', str(self.args.cache_dir),
                '--log-dir', str(self.args.log_dir), '--model-dir', str(self.args.model_dir)]
        memory = SimpleNamespace(virtual_memory=lambda: SimpleNamespace(total=4 * 1024**3))
        with patch.dict(sys.modules, {'psutil': memory}), patch('diarize.load_credentials'), patch('diarize.preflight'), patch('diarize.version', return_value='3.8.6'), patch('diarize.download', side_effect=self.download), patch('diarize.child', side_effect=self.stage) as child, redirect_stdout(io.StringIO()):
            with patch.object(diarize, 'ENGINE_VERSION', 1):
                self.assertEqual(diarize.main(argv), 0)
            self.assertEqual(child.call_count, 3)
            self.assertEqual(diarize.main(argv), 0)
            self.assertEqual(child.call_count, 6)
            self.assertEqual(diarize.main(argv), 0)
            self.assertEqual(child.call_count, 6)
        for call in child.call_args_list:
            config = json.loads(Path(call.args[0][2]).read_text(encoding='utf-8'))
            self.assertIsNone(config['language'])

    def test_success_then_skip_without_download_or_model(self):
        with patch('diarize.download', side_effect=self.download) as download, patch('diarize.child', side_effect=self.stage) as child:
            first = diarize.run_job(self.url, self.args, threading.Semaphore(1), 'settings1')
            second = diarize.run_job(self.url, self.args, threading.Semaphore(1), 'settings1')
        self.assertEqual(first['status'], 'completed')
        self.assertEqual(second['status'], 'skipped')
        self.assertEqual(download.call_count, 1)
        self.assertEqual(child.call_count, 3)
        self.assertTrue(Path(first['output']).read_text(encoding='utf-8').startswith('Real title'))

    def test_failed_alignment_leaves_no_final_transcript_and_resumes(self):
        def fail_alignment(command, log, label, **kwargs):
            if command[3] == 'align':
                raise diarize.JobError('simulated alignment failure')
            self.stage(command, log, label)
        with patch('diarize.download', side_effect=self.download), patch('diarize.child', side_effect=fail_alignment):
            first = diarize.run_job(self.url, self.args, threading.Semaphore(1), 'settings1')
        self.assertEqual(first['status'], 'failed')
        self.assertFalse(list(self.args.out.glob('*.txt')))
        with patch('diarize.download', side_effect=self.download), patch('diarize.child', side_effect=self.stage) as child:
            second = diarize.run_job(self.url, self.args, threading.Semaphore(1), 'settings1')
        self.assertEqual(second['status'], 'completed')
        self.assertEqual([call.args[0][3] for call in child.call_args_list], ['align', 'diarize'])

    def test_settings_change_invalidates_completed_state(self):
        with patch('diarize.download', side_effect=self.download), patch('diarize.child', side_effect=self.stage):
            diarize.run_job(self.url, self.args, threading.Semaphore(1), 'settings1')
        self.assertIsNone(diarize.completed_path(self.args.out, diarize.key_for(self.url), 'settings2'))

    def test_export_completed_checkpoint_without_audio_network_or_model_slot(self):
        work = self.args.cache_dir / diarize.key_for(self.url)
        diarize.atomic_json(work / 'video.json', {'title': 'Recovered', 'url': self.url})
        diarize.atomic_json(work / 'settings1' / 'diarized.json', {'language': 'en', 'segments': [{'start': 1, 'end': 2, 'text': 'saved', 'speaker': 'A'}]})
        with patch('diarize.download') as download, patch('diarize.child') as child:
            result = diarize.run_job(self.url, self.args, threading.Semaphore(0), 'settings1')
        self.assertEqual(result['status'], 'completed')
        download.assert_not_called()
        child.assert_not_called()
        self.assertIn('saved', Path(result['output']).read_text(encoding='utf-8'))

    def test_stalled_child_is_terminated_and_original_log_preserved(self):
        log = self.root / 'stall.log'
        log.write_text('previous failure\n', encoding='utf-8')
        with self.assertRaisesRegex(diarize.JobError, 'no log activity'):
            diarize.child([sys.executable, '-c', 'import time; time.sleep(60)'], log, 'stall test', stall_seconds=0.1)
        self.assertIn('previous failure', log.read_text(encoding='utf-8'))
        self.assertIn('exit', log.read_text(encoding='utf-8'))

    def test_failed_force_invalidates_old_checkpoints_and_completion(self):
        with patch('diarize.download', side_effect=self.download), patch('diarize.child', side_effect=self.stage):
            diarize.run_job(self.url, self.args, threading.Semaphore(1), 'settings1')
        self.args.force = True
        with patch('diarize.download', side_effect=self.download), patch('diarize.child', side_effect=diarize.JobError('failed forced run')):
            failed = diarize.run_job(self.url, self.args, threading.Semaphore(1), 'settings1')
        self.assertEqual(failed['status'], 'failed')
        self.assertIsNone(diarize.completed_path(self.args.out, diarize.key_for(self.url), 'settings1'))
        stages = self.args.cache_dir / diarize.key_for(self.url) / 'settings1'
        self.assertFalse((stages / 'transcribed.json').exists())
        self.assertFalse((stages / 'diarized.json').exists())

    def test_forced_title_change_replaces_previous_filename(self):
        with patch('diarize.download', side_effect=self.download), patch('diarize.child', side_effect=self.stage):
            first = diarize.run_job(self.url, self.args, threading.Semaphore(1), 'settings1')
        self.args.force = True
        def renamed(url, work, args, job_id):
            return {**self.download(url, work, args, job_id), 'title': 'Updated actual title'}
        with patch('diarize.download', side_effect=renamed), patch('diarize.child', side_effect=self.stage):
            second = diarize.run_job(self.url, self.args, threading.Semaphore(1), 'settings1')
        self.assertEqual(second['status'], 'completed')
        self.assertFalse(Path(first['output']).exists())
        self.assertTrue(Path(second['output']).exists())
        self.assertEqual(len(list(self.args.out.glob('*.txt'))), 1)

    def test_state_cannot_point_outside_output_directory(self):
        diarize.atomic_json(self.args.out / '.state' / 'id.json', {'fingerprint': 'settings1', 'filename': '../escape.txt'})
        self.assertIsNone(diarize.completed_path(self.args.out, 'id', 'settings1'))

    def test_lock_rejects_concurrent_writer_then_releases(self):
        with diarize.job_lock(self.root / 'job'):
            with self.assertRaises(diarize.JobError):
                with diarize.job_lock(self.root / 'job'):
                    self.fail('Second lock was acquired')
        with diarize.job_lock(self.root / 'job'):
            pass


if __name__ == '__main__':
    unittest.main()
