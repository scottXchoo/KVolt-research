"""Natural generation preparation for the actual frozen session inputs.

This ledger describes only visited shapes. An EOS never licenses unvisited
Decode kernels, and these forwards do not contribute request timing or energy.
"""
from ..config import GROUPS
from ..contracts import CacheFormat, RequestSpec, generation_stop
from ..execution.contracts import ExecutionError, error_record


def _receipt(receipt, stage):
    if not isinstance(receipt, dict) or receipt.get('verified') is not True:
        raise ExecutionError('GPU_RECEIPT_UNVERIFIED', stage, 'warmup device receipt is unverified', receipt=receipt)
    return receipt


def _shape(length):
    return {'cache_length': length, 'key_residual_length': length % 128,
            'value_residual_length': min(length, 128)}


def formal_warmup(*, backend, device, prepared, snapshot, clock, stop):
    """Warm each distinct frozen input/context to its largest actual cap.

    Identical input IDs alone do not merge different prompt bytes or metadata.
    Repeated requests of the same context share preparation at the maximum cap.
    No controller/LUT prediction or forced EOS suppression occurs here.
    """
    report = {'policy': 'formal', 'formal_coverage_qualified': False,
              'coverage_scope': 'actual_inputs_natural_reachable_shapes',
              'started_ns': clock(), 'cases': [], 'errors': [], 'interrupted': False}
    contexts = {}
    for item in prepared.requests:
        key = (item.spec.token_ids, item.spec.prompt_sha256, item.metadata_json)
        context = contexts.setdefault(key, {'item': item, 'nmax': 1, 'requests': []})
        context['nmax'] = max(context['nmax'], item.spec.nmax)
        context['requests'].append({'request_id': item.spec.request_id, 'nmax': item.spec.nmax})

    def boundary():
        if stop.requested:
            report['interrupted'] = True
            raise ExecutionError('WARMUP_INTERRUPTED', 'warmup', 'formal warmup interrupted')

    try:
        boundary()
        report['prepare'] = _receipt(device.prepare_request(), 'warmup_preparation')
        boundary()
        report['frequency'] = _receipt(device.apply(1122), 'warmup_frequency')
        boundary()
        for context in contexts.values():
            item, nmax = context['item'], context['nmax']
            for group in GROUPS if nmax > 1 else (None,):
                boundary()
                case = {'length': item.spec.length, 'group': group, 'nmax': nmax,
                        'requests': context['requests'], 'metadata': item.metadata,
                        'input_token_sha256': item.spec.token_sha256,
                        'prompt_sha256': item.spec.prompt_sha256,
                        'started_ns': clock(), 'prefill_forwards': 0,
                        'initial_packs': 0, 'decode_forwards': 0, 'token_ids': [],
                        'n_output_tokens': 0, 'stop_reason': None, 'status': 'failed',
                        'cache_released': False, 'final_cache_length': None,
                        'decode_shapes': [], 'initial_pack_shape': None,
                        'mean_tpot_s': None, 'decode_j_per_token': None}
                report['cases'].append(case)
                cache = None
                try:
                    spec = RequestSpec(f'formal-warmup-{len(report["cases"])}',
                                       item.spec.token_ids, item.spec.prompt_sha256, nmax)
                    first = backend.prefill_raw(spec)
                    cache = first.cache
                    case['prefill_forwards'] = 1
                    if (cache.metadata.length != spec.length or cache.metadata.request_id != spec.request_id
                            or cache.format is not CacheFormat.RAW
                            or type(first.ready_ns) is not int or first.ready_ns <= case['started_ns']):
                        raise ExecutionError('BACKEND_CACHE_CONTRACT', 'warmup', 'invalid warmup Prefill cache')
                    case['token_ids'].append(first.token_id)
                    previous_ns = first.ready_ns
                    reason = generation_stop(tuple(case['token_ids']), nmax)
                    boundary()
                    if reason is None:
                        case['initial_pack_shape'] = _shape(cache.metadata.length)
                        backend.pack_initial(cache, group)
                        if (cache.format is not CacheFormat.PACKED or cache.metadata.length != spec.length
                                or cache.metadata.group_size != group):
                            raise ExecutionError('BACKEND_CACHE_CONTRACT', 'warmup', 'invalid warmup initial packing')
                        case['initial_packs'] = 1
                        boundary()
                        while reason is None:
                            boundary()
                            shape = _shape(cache.metadata.length)
                            token = backend.decode_one(cache, case['token_ids'][-1])
                            if (token.cache is not cache or cache.metadata.length != spec.length + len(case['token_ids'])
                                    or type(token.ready_ns) is not int or token.ready_ns <= previous_ns):
                                raise ExecutionError('BACKEND_CACHE_CONTRACT', 'warmup', 'invalid warmup Decode cache/clock')
                            case['decode_forwards'] += 1
                            case['decode_shapes'].append({**shape, 'result_cache_length': cache.metadata.length})
                            case['token_ids'].append(token.token_id)
                            previous_ns = token.ready_ns
                            reason = generation_stop(tuple(case['token_ids']), nmax)
                            boundary()
                    case['stop_reason'] = reason.value
                    case['skip_reason'] = (('first_token_eos' if reason.value == 'eos' else 'nmax_one')
                                           if case['initial_packs'] == 0 else None)
                    case['status'] = 'completed'
                except Exception as error:
                    case['status'] = 'interrupted' if stop.requested else 'failed'
                    case['stop_reason'] = 'interrupted' if stop.requested else None
                    report['errors'].append(error_record(error, stage='warmup_forward'))
                finally:
                    case['n_output_tokens'] = len(case['token_ids'])
                    if cache is not None:
                        try:
                            if cache.format is not CacheFormat.RELEASED:
                                case['final_cache_length'] = cache.metadata.length
                            backend.release_request(cache)
                            case['cache_released'] = cache.format is CacheFormat.RELEASED
                            if not case['cache_released']:
                                raise ExecutionError('CACHE_RELEASE_FAILED', 'warmup_release', 'warmup cache remains live')
                        except Exception as error:
                            report['errors'].append(error_record(error, stage='warmup_release'))
                    case['ended_ns'] = clock()
                if report['errors']:
                    break
            if report['errors']:
                break
    except Exception as error:
        report['errors'].append(error_record(error, stage='warmup_preparation'))
    finally:
        try:
            report['restore'] = _receipt(device.restore(snapshot), 'warmup_restore')
        except Exception as error:
            report['errors'].append(error_record(error, stage='warmup_restore'))
        report['ended_ns'] = clock()
    # A signal delivered by the final release/restore must also invalidate the
    # ledger, even when there is no subsequent forward boundary to observe it.
    report['interrupted'] = report['interrupted'] or stop.requested
    report['formal_coverage_qualified'] = not report['errors'] and not report['interrupted']
    if not report['formal_coverage_qualified']:
        raise ExecutionError('WARMUP_FAILED', 'warmup', 'formal warmup or cleanup failed', receipt=report)
    return report
