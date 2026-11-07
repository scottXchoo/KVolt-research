"""Sequential adaptive requests with explicit measurement and cleanup boundaries.

The CLI acquires the hardware lock before constructing/loading a real Backend.
This orchestrator is dependency-injected so CPU tests can inspect every boundary.
It never changes the PC/DC policy or retains Tensor aliases outside CacheHandle.
"""
from dataclasses import asdict
import hashlib
import time
import uuid

from ..config import GROUPS, RuntimeConfig
from ..contracts import CacheFormat, RequestSpec, generation_stop
from ..errors import ContractError
from ..policy.prefill import PrefillController
from ..recording import request_record, to_json_native
from .contracts import (
    EXIT_INPUT, EXIT_OK, EXIT_PREPARATION, EXIT_TECHNICAL,
    ExecutionError, PreparationError, StopController, error_record,
)


def _no_energy(reason='not_collected'):
    return {'status': reason, 'valid': False, 'reason': reason,
            'prefill_energy_j': None, 'decode_energy_j': None,
            'total_energy_j': None, 'decode_j_per_token': None, 'prefix': None}


def _invalidate_energy(energy):
    """Retain diagnostics without publishing normal energy from a failed trace."""
    fields = ('prefill_energy_j', 'decode_energy_j', 'total_energy_j', 'decode_j_per_token')
    if energy.get('valid'):
        energy['prefix'] = {name: energy.get(name) for name in (*fields, 'rails')}
    for name in fields:
        energy[name] = None
    reason = energy.get('reason')
    energy.update(status='invalid', valid=False, rails={},
                  reason=reason if reason not in (None, 'not_collected') else 'collector_failure')


def _new_result(prepared, index):
    return {'schema_version': 'kvolt-request-result-v1',
            'request': request_record(prepared, index=index),
            'generation': {'status': 'not_run', 'token_ids': [], 'token_ready_ns': [],
                           'n_output_tokens': 0, 'stop_reason': None,
                           'prefill_forwards': 0, 'initial_packs': 0, 'decode_forwards': 0,
                           'stage_timing': {'initial_pack': None, 'first_decode': None},
                           'final_cache_length': None},
            'timing': {'status': 'not_run', 'valid': False, 'reason': 'not_run',
                       't0_ns': None, 't1_ns': None, 'tN_ns': None,
                       'ttft_s': None, 'mean_tpot_s': None},
            'energy': _no_energy('not_run'),
            'decisions': {'prefill': None, 'decode': None},
            'device': {'start': None, 'prefill': None, 'decode': None, 'restore': None},
            'cleanup': {'status': 'not_run', 'cache_released': None, 'gpu_restored': None},
            'errors': [], 'events': [], 'paths': {}}


class _RequestInterrupted(Exception):
    pass


class SessionRunner:
    def __init__(self, *, backend, repo, device, recorder, decode_controller,
                 clock=time.monotonic_ns, stop=None, collector_factory=None,
                 emit=None, warmup=True, qualification=None):
        self.backend, self.repo, self.device, self.recorder = backend, repo, device, recorder
        self.dc, self.clock = decode_controller, clock
        self.pc = PrefillController(repo, clock=clock)
        self.stop = stop if stop is not None else StopController()
        self.collector_factory, self.emit = collector_factory, emit or (lambda record: None)
        self.warmup = warmup
        self.qualification = qualification
        self._first_collector = None

    def _performance_gate(self, prepared, options, provenance):
        """Reject incomplete or mismatched qualification before hardware work."""
        from ..qualification.contracts import QualificationGrant
        if not isinstance(self.qualification, QualificationGrant):
            raise ContractError('M4 validated qualification grant is required for performance execution')
        if self.warmup != 'formal' or self.collector_factory is None or not options.collect_power:
            raise ContractError('performance execution requires formal warmup and a power collector')
        grant = self.qualification
        grant.assert_requests(prepared)
        binding = grant.binding
        source = self.repo.provenance
        lut = {name: getattr(source, name) for name in
               ('kind', 'source_device', 'source_commit', 'manifest_sha256', 'files_sha256')}
        lut['files_sha256'] = dict(lut['files_sha256'])
        if source.kind != 'measured' or lut != binding.get('lut'):
            raise ContractError('runtime LUT differs from measured qualification source')
        if options.native_profile != binding.get('native_profile'):
            raise ContractError('runtime native profile differs from qualification')
        profile = getattr(self.device, 'profile', None)
        if profile is None or profile.as_dict() != binding.get('device_profile'):
            raise ContractError('runtime device settings differ from qualification')
        for field in ('device_profile', 'power_profile'):
            if field in provenance and provenance[field] != binding.get(field):
                raise ContractError('runtime ' + field + ' differs from qualification')
        # Construct without starting acquisition, retaining this exact collector
        # for request zero rather than trusting a caller-provided provenance dict.
        self._first_collector = self.collector_factory(0)
        self._check_collector(self._first_collector)

    def _check_collector(self, collector):
        profile = getattr(collector, 'profile', None)
        if profile is None or profile.to_dict() != self.qualification.binding.get('power_profile'):
            raise ContractError('runtime power settings differ from qualification')

    def _interruption_boundary(self):
        if self.stop.requested:
            raise _RequestInterrupted()

    @staticmethod
    def _verified_device_receipt(receipt, stage):
        if not isinstance(receipt, dict) or receipt.get('verified') is not True:
            raise ExecutionError('GPU_RECEIPT_UNVERIFIED', stage,
                                 'device preparation/frequency receipt is not verified', receipt=receipt)
        return receipt

    def _warmup(self, prepared, snapshot):
        """Development warmup, outside request intervals and never M4 evidence.

        Use frozen real input IDs, all five Gs for every distinct active length.
        A natural first EOS skips packing/Decode, including during warmup. This
        limited warmup does not claim complete JIT/residual-shape qualification.
        """
        if self.warmup == 'formal':
            from ..qualification.warmup import formal_warmup
            return formal_warmup(backend=self.backend, device=self.device, prepared=prepared,
                                 snapshot=snapshot, clock=self.clock, stop=self.stop)
        report = {'policy': 'development_basic' if self.warmup else 'none',
                  'formal_coverage_qualified': False, 'cases': [], 'errors': [], 'started_ns': self.clock()}
        if not self.warmup:
            report['ended_ns'] = self.clock()
            return report
        shapes = {}
        for item in prepared.requests:
            key = (item.spec.length, item.spec.nmax > 1)
            shapes.setdefault(key, item)
        try:
            if self.stop.requested:
                report['interrupted'] = True
            else:
                report['prepare'] = self._verified_device_receipt(self.device.prepare_request(), 'warmup_preparation')
                report['frequency'] = self._verified_device_receipt(self.device.apply(1122), 'warmup_frequency')
            for (length, active), item in shapes.items():
                for group in GROUPS if active else (None,):
                    if self.stop.requested:
                        report['interrupted'] = True
                        break
                    cache = None
                    case = {'length': length, 'group': group, 'input_token_sha256': item.spec.token_sha256,
                            'started_ns': self.clock(), 'initial_packs': 0, 'decode_forwards': 0}
                    report['cases'].append(case)
                    try:
                        spec = RequestSpec(f'warmup-{len(report["cases"])}', item.spec.token_ids,
                                           item.spec.prompt_sha256, 2 if active else 1)
                        token = self.backend.prefill_raw(spec)
                        cache = token.cache
                        case['first_token_id'] = token.token_id
                        stop = generation_stop((token.token_id,), spec.nmax)
                        if stop is None and not self.stop.requested:
                            self.backend.pack_initial(cache, group)
                            case['initial_packs'] = 1
                            if self.stop.requested:
                                case['skip_reason'] = 'interrupted'
                                report['interrupted'] = True
                            else:
                                following = self.backend.decode_one(cache, token.token_id)
                                case['decode_forwards'] = 1
                                case['second_token_id'] = following.token_id
                        else:
                            case['skip_reason'] = stop.value if stop else 'interrupted'
                    except Exception as error:
                        report['errors'].append(error_record(error, stage='warmup_forward'))
                    finally:
                        if cache is not None:
                            try:
                                self.backend.release_request(cache)
                            except Exception as error:
                                report['errors'].append(error_record(error, stage='warmup_release'))
                        case['ended_ns'] = self.clock()
                    if report['errors']:
                        break
                if report['errors'] or self.stop.requested:
                    break
        except Exception as error:
            report['errors'].append(error_record(error, stage='warmup_preparation'))
        finally:
            try:
                report['restore'] = self.device.restore(snapshot)
                if not report['restore'].get('verified'):
                    raise ExecutionError('GPU_RESTORE_FAILED', 'warmup_restore', 'warmup GPU restoration is unverified')
            except Exception as error:
                report['errors'].append(error_record(error, stage='warmup_restore'))
            report['ended_ns'] = self.clock()
        if report['errors']:
            raise ExecutionError('WARMUP_FAILED', 'warmup', 'development warmup or cleanup failed', receipt=report)
        return report

    def _check_token(self, token, cache, expected_length, previous_ns):
        if cache is not token.cache or cache.metadata.length != expected_length:
            raise ExecutionError('BACKEND_CACHE_CONTRACT', 'backend', 'token/cache identity or length mismatch')
        if type(token.ready_ns) is not int or token.ready_ns <= previous_ns:
            raise ExecutionError('TOKEN_CLOCK_CONTRACT', 'backend', 'token-ready clock is not strictly increasing')

    def _request(self, item, index, snapshot, collect_power):
        from ..telemetry import energy_metrics, timing_metrics
        result = _new_result(item, index)
        generation, errors = result['generation'], result['errors']
        cache = collector = None
        pc = dc = None
        t0 = None
        stage = 'request_preparation'
        result['attempt_started_ns'] = self.clock()
        generation['status'] = 'failed'
        try:
            if collect_power:
                if self.collector_factory is None:
                    raise ExecutionError('POWER_COLLECTOR_MISSING', 'power_start', 'power collection requested without a collector')
                collector = self._first_collector if index == 0 and self._first_collector is not None else self.collector_factory(index)
                if self.qualification is not None:
                    self._check_collector(collector)
            result['device']['start'] = self._verified_device_receipt(self.device.prepare_request(), stage)
            if collect_power:
                stage = 'power_start'
                result['power_start'] = collector.start()
                if not result['power_start'].get('ready') or result['power_start'].get('errors'):
                    raise ExecutionError('POWER_NOT_READY', stage, 'collector did not produce a valid pre-t0 sample',
                                         receipt=result['power_start'])
            if self.stop.requested:
                generation.update(status='interrupted', stop_reason='interrupted')
            else:
                # No file I/O or precomputed per-request decisions in this interval.
                stage = 'prefill_controller'
                t0 = self.clock()
                pc = self.pc.choose(item.spec.length, t0)
                self._interruption_boundary()
                stage = 'prefill_frequency'
                result['device']['prefill'] = self._verified_device_receipt(
                    self.device.apply(pc.selected.frequency_mhz), stage)
                self._interruption_boundary()
                stage = 'prefill'
                first = self.backend.prefill_raw(item.spec)
                cache = first.cache
                self._check_token(first, cache, item.spec.length, t0)
                generation['prefill_forwards'] = 1
                generation['token_ids'].append(first.token_id)
                generation['token_ready_ns'].append(first.ready_ns)
                stop = generation_stop(tuple(generation['token_ids']), item.spec.nmax)
                if stop is None and self.stop.requested:
                    generation.update(status='interrupted', stop_reason='interrupted')
                else:
                    # DC itself owns the first-EOS/Nmax=1 skip contract. It performs
                    # no Q/LUT/clock work in that branch and no packing follows it.
                    stage = 'decode_controller'
                    dc = self.dc.choose(item.spec.length, item.spec.nmax, first.token_id, first.ready_ns)
                    if stop is None:
                        if dc.status == 'skipped' or dc.selected is None:
                            raise ExecutionError('DC_CONTRACT', stage, 'active Decode has no selected setting')
                        if self.stop.requested:
                            generation.update(status='interrupted', stop_reason='interrupted')
                        else:
                            stage = 'decode_frequency'
                            result['device']['decode'] = self._verified_device_receipt(
                                self.device.apply(dc.selected.frequency_mhz), stage)
                            if not self.stop.requested:
                                stage = 'initial_pack'
                                pack_timing = {'start_ns': self.clock(), 'end_ns': None}
                                generation['stage_timing']['initial_pack'] = pack_timing
                                try:
                                    self.backend.pack_initial(cache, dc.selected.group)
                                finally:
                                    pack_timing['end_ns'] = self.clock()
                                generation['initial_packs'] = 1
                                while stop is None and not self.stop.requested:
                                    stage = 'decode'
                                    first_decode_timing = None
                                    if generation['decode_forwards'] == 0:
                                        first_decode_timing = {'start_ns': self.clock(), 'end_ns': None}
                                        generation['stage_timing']['first_decode'] = first_decode_timing
                                    try:
                                        token = self.backend.decode_one(cache, generation['token_ids'][-1])
                                    finally:
                                        if first_decode_timing is not None:
                                            first_decode_timing['end_ns'] = self.clock()
                                    self._check_token(token, cache, item.spec.length + len(generation['token_ids']),
                                                      generation['token_ready_ns'][-1])
                                    generation['decode_forwards'] += 1
                                    generation['token_ids'].append(token.token_id)
                                    generation['token_ready_ns'].append(token.ready_ns)
                                    stop = generation_stop(tuple(generation['token_ids']), item.spec.nmax)
                            if stop is None:
                                generation.update(status='interrupted', stop_reason='interrupted')
                    if stop is not None:
                        generation.update(status='completed', stop_reason=stop.value)
        except _RequestInterrupted:
            generation.update(status='interrupted', stop_reason='interrupted')
        except Exception as error:
            errors.append(error_record(error, stage=stage))
            result['failed_ns'] = self.clock()
        finally:
            generation['n_output_tokens'] = len(generation['token_ids'])
            completed = generation['status'] == 'completed'
            try:
                # Preserve selected decisions even when generation fails, but
                # build their nested evidence only after the last token boundary.
                result['decisions']['prefill'] = asdict(pc) if pc is not None else None
                result['decisions']['decode'] = asdict(dc) if dc is not None else None
                if dc is not None and dc.knee is not None and dc.knee.status == 'error_recovered':
                    result['events'].append({'kind': 'knee_error_recovered',
                                             **result['decisions']['decode']['knee']})
                if cache is not None and cache.format is not CacheFormat.RELEASED:
                    generation['final_cache_length'] = cache.metadata.length
                result['timing'] = timing_metrics(t0, generation['token_ready_ns'], completed=completed)
                result['timing']['status'] = 'valid' if result['timing']['valid'] else 'invalid'
                if completed and not result['timing']['valid']:
                    errors.append(error_record(ExecutionError('TIMING_INVALID', 'measurement',
                        result['timing'].get('reason') or 'invalid token timing')))
                pc, dc = result['decisions']['prefill'], result['decisions']['decode']
                result['timing'].update(tp_ns=pc['tp_ns'] if pc else None,
                                        td_ns=dc['td_ns'] if dc else None)
                result['slo'] = {'ttft_met': (result['timing']['ttft_s'] <= pc['target_s'])
                                 if pc and result['timing']['valid'] else None,
                                 'mean_tpot_met': (result['timing']['mean_tpot_s'] <= dc['target_s'])
                                 if dc and dc['target_s'] is not None and result['timing']['valid']
                                 and result['timing']['mean_tpot_s'] is not None else None}
            except Exception as error:
                # Metrics and evidence failures must never bypass resource cleanup.
                errors.append(error_record(error, stage='timing_measurement'))
                times = generation['token_ready_ns']
                result['timing'] = {'status': 'invalid', 'valid': False, 'reason': 'measurement_failure',
                                    't0_ns': t0, 't1_ns': times[0] if times else None,
                                    'tN_ns': times[-1] if times else None,
                                    'ttft_s': None, 'mean_tpot_s': None, 'prefix': None}
                result['slo'] = {'ttft_met': None, 'mean_tpot_met': None}
            result['energy'] = _no_energy()
            cleanup_errors = []
            if collector is not None:
                try:
                    times = generation['token_ready_ns']
                    receipt = collector.finish(times[-1]) if times else collector.stop()
                    result['power'] = {key: value for key, value in receipt.items()
                                       if key not in ('samples', 'raw_trace_jsonl')}
                    samples = receipt.get('samples', [])
                    raw = receipt.get('raw_trace_jsonl')
                    if raw is not None:
                        raw_path = f'telemetry/{index:06d}/power.raw.jsonl'
                        self.recorder.write_text(raw_path, raw)
                        result['power']['raw_path'] = self.recorder.path(raw_path)
                        result['power']['raw_sha256'] = hashlib.sha256(raw.encode('utf-8')).hexdigest()
                    from ..telemetry import samples_csv
                    csv_text = samples_csv(samples)
                    trace_path = f'telemetry/{index:06d}/power.csv'
                    self.recorder.write_text(trace_path, csv_text)
                    result['power']['csv_path'] = self.recorder.path(trace_path)
                    result['power']['csv_sha256'] = hashlib.sha256(csv_text.encode('utf-8')).hexdigest()
                    profile = receipt.get('profile', {})
                    result['energy'] = energy_metrics(samples, t0, times[0] if times else None,
                        times[-1] if times else None, len(times), completed=completed,
                        max_gap_ns=profile['max_gap_ns'])
                    result['energy']['status'] = 'valid' if result['energy']['valid'] else 'invalid'
                    if receipt.get('errors') or (completed and not result['energy']['valid']):
                        raise ExecutionError('POWER_MEASUREMENT_FAILED', 'measurement',
                                             'power collection or trace validity failed', receipt=result['power'])
                except Exception as error:
                    errors.append(error_record(error, stage='measurement'))
                    _invalidate_energy(result['energy'])
                finally:
                    try:
                        stopped = collector.stop()
                        if stopped.get('errors') and not result.get('power', {}).get('errors'):
                            cleanup_errors.append(error_record(ExecutionError(
                                'POWER_STOP_FAILED', 'cleanup', str(stopped['errors']))))
                    except Exception as error:
                        cleanup_errors.append(error_record(error, stage='power_cleanup'))
            result['cleanup_started_ns'] = self.clock()
            try:
                if cache is not None:
                    self.backend.release_request(cache)
                result['cleanup']['cache_released'] = True
            except Exception as error:
                cleanup_errors.append(error_record(error, stage='cache_release'))
                result['cleanup']['cache_released'] = False
            try:
                result['device']['restore'] = self.device.restore(snapshot)
                result['cleanup']['gpu_restored'] = bool(result['device']['restore'].get('verified'))
                if not result['cleanup']['gpu_restored']:
                    raise ExecutionError('GPU_RESTORE_FAILED', 'cleanup', 'GPU restoration receipt is unverified')
            except Exception as error:
                result['cleanup']['gpu_restored'] = False
                cleanup_errors.append(error_record(error, stage='gpu_restore'))
            result['cleanup'].update(status='failed' if cleanup_errors else 'completed', errors=cleanup_errors)
            result['cleanup_ended_ns'] = self.clock()
            # A completed normal timing/energy result remains valid if only
            # restoration fails. No second duration or energy interval is added.
            errors.extend(cleanup_errors)
        return result

    def _save_emit(self, result, item, index, session):
        tokens = result['generation']['token_ids']
        text = None
        if tokens:
            try:
                text = self.backend.tokenizer.decode(tokens, skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=False)
            except Exception as error:
                result['errors'].append(error_record(error, stage='output_decode'))
        result['output_text'] = text
        paths = self.recorder.save_request(index, item, result, result['generation'], text)
        result['paths'] = paths
        # Store paths in the final result after all files have actually appeared.
        self.recorder.atomic_json(f'requests/{index:06d}/result.json', result)
        record = {'schema_version': 'kvolt-stdout-v1', 'record_type': 'request_result',
                  'session_id': session['session_id'], 'request_id': item.spec.request_id,
                  'attempt': 1, 'index': index, 'text': text,
                  'generation_status': result['generation']['status'],
                  'timing_status': result['timing']['status'], 'energy_status': result['energy']['status'],
                  'cleanup_status': result['cleanup']['status'],
                  'artifact_root': str(self.recorder.output_dir), **paths}
        self.emit(record)
        session['request_records'].append(record)

    def run(self, prepared, options, *, provenance=None, session_id=None):
        session = {'schema_version': 'kvolt-session-v1', 'session_id': session_id or uuid.uuid4().hex,
                   'status': 'failed', 'mode': options.mode, 'collect_power': options.collect_power,
                   'started_ns': self.clock(), 'provenance': provenance or {}, 'errors': [],
                   'request_records': [], 'results': [], 'exit_code': EXIT_OK}
        snapshot = None
        stage = 'device_preparation'
        try:
            if options.mode == 'performance':
                stage = 'qualification'
                self._performance_gate(prepared, options, provenance or {})
                session['qualification'] = self.qualification.as_dict()
            stage = 'device_preparation'
            snapshot = self.device.capture()
            if not snapshot.get('verified'):
                raise PreparationError('DEVICE_CAPTURE_UNVERIFIED', stage, 'device capture is unverified')
            if options.mode == 'performance':
                self.qualification.assert_device(snapshot)
            stage = 'device_snapshot_save'
            self.recorder.atomic_json('device.before.json', snapshot)
            stage = 'device_preparation'
            session['device_preflight'] = self.device.verify_prepared()
            if not session['device_preflight'].get('verified'):
                raise PreparationError('DEVICE_NOT_PREPARED', stage, 'device preparation is unverified')
            if self.stop.requested:
                session['exit_code'] = self.stop.exit_code
            else:
                stage = 'model_load'
                self.backend.load_session(options.model_dir, RuntimeConfig())
                session['backend'] = to_json_native(self.backend.provenance)
                # CPU preflight uses the pinned Rust tokenizer without importing
                # torch. Cross-check the loaded HF tokenizer before timed work.
                stage = 'tokenizer_consistency'
                for item in prepared.requests:
                    ids = tuple(self.backend.tokenizer(item.prompt.text, add_special_tokens=False)['input_ids'])
                    if ids != item.spec.token_ids:
                        raise ExecutionError('TOKENIZER_MISMATCH', stage, 'loaded tokenizer differs from frozen input IDs')
                stage = 'warmup'
                session['warmup'] = self._warmup(prepared, snapshot)
                if options.mode == 'performance':
                    session['provenance']['formal_performance_qualified'] = True
                for index, item in enumerate(prepared.requests):
                    if session['exit_code'] or self.stop.requested:
                        break
                    stage = 'request'
                    result = self._request(item, index, snapshot, options.collect_power)
                    session['results'].append(result)
                    try:
                        self._save_emit(result, item, index, session)
                    except Exception as error:
                        session['errors'].append(error_record(error, stage='result_save_or_stdout'))
                        session['exit_code'] = EXIT_TECHNICAL
                    if result['errors']:
                        session['exit_code'] = EXIT_TECHNICAL
                    if self.stop.requested:
                        session['exit_code'] = self.stop.exit_code
        except Exception as error:
            if stage == 'warmup' and isinstance(getattr(error, 'receipt', None), dict):
                session['warmup'] = error.receipt
            session['errors'].append(error_record(error, stage=stage))
            if isinstance(error, ContractError) and options.mode == 'performance':
                session['exit_code'] = EXIT_INPUT
            elif isinstance(error, PreparationError) or stage == 'device_preparation':
                session['exit_code'] = EXIT_PREPARATION
            else:
                session['exit_code'] = EXIT_TECHNICAL
            # Warmup may have touched GPU settings. Retry restoration only as
            # cleanup, never repeat a failed request or run a different setting.
            if snapshot is not None and stage in ('model_load', 'tokenizer_consistency', 'warmup'):
                try:
                    session['session_restore'] = self.device.restore(snapshot)
                    if not session['session_restore'].get('verified'):
                        raise ExecutionError('GPU_RESTORE_FAILED', 'session_restore', 'session GPU restoration is unverified')
                except Exception as cleanup:
                    session['errors'].append(error_record(cleanup, stage='session_restore'))
        finally:
            if self.stop.requested:
                session['exit_code'] = self.stop.exit_code
            for index in range(len(session['results']), len(prepared.requests)):
                item = prepared.requests[index]
                result = _new_result(item, index)
                result['not_run_reason'] = 'session_interrupted' if self.stop.requested else 'session_aborted'
                session['results'].append(result)
                try:
                    self._save_emit(result, item, index, session)
                except Exception as error:
                    session['errors'].append(error_record(error, stage='not_run_save_or_stdout'))
                    if not self.stop.requested:
                        session['exit_code'] = EXIT_TECHNICAL
            counts = {'generation_completed': 0, 'generation_failed': 0, 'generation_interrupted': 0,
                      'not_run': 0, 'timing_valid': 0, 'energy_valid': 0,
                      'energy_not_collected': 0, 'request_cleanup_failed': 0}
            for result in session['results']:
                status = result['generation']['status']
                counts['not_run' if status == 'not_run' else 'generation_' + status] += 1
                counts['timing_valid'] += int(result['timing']['valid'])
                counts['energy_valid'] += int(result['energy']['valid'])
                counts['energy_not_collected'] += int(result['energy']['status'] == 'not_collected')
                counts['request_cleanup_failed'] += int(result['cleanup']['status'] == 'failed')
            session['counts'] = counts
            session['status'] = 'interrupted' if self.stop.requested else ('completed' if session['exit_code'] == 0 else 'failed')
            if session['status'] != 'completed':
                session['provenance']['formal_performance_qualified'] = False
            session['ended_ns'] = self.clock()
            try:
                self.recorder.save_session(session)
            except Exception as error:
                session['errors'].append(error_record(error, stage='session_save'))
                if not self.stop.requested:
                    session.update(status='failed', exit_code=EXIT_TECHNICAL)
                session['provenance']['formal_performance_qualified'] = False
            self.emit({'schema_version': 'kvolt-stdout-v1', 'record_type': 'session_result',
                       'session_id': session['session_id'], 'status': session['status'],
                       'counts': session['counts'], 'errors': session['errors'],
                       'exit_code': session['exit_code'], 'artifact_root': str(self.recorder.output_dir),
                       'session_path': self.recorder.session_path})
        return session
