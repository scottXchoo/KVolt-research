"""Sequential generation-only lm-eval adapter for the project's KIVI loader.

Importing this optional module requires lm-eval, but neither imports its HF
backend nor loads Torch, Transformers, model weights, or power sensors.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from model import kivi_llama


MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


def _positive_int(name, value):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _dump(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _diagnostic_error(original, stage, error):
    message = f"lm-eval {stage} failed: {type(error).__name__}: {error}"
    if hasattr(original, "add_note"):
        original.add_note(message)
    print(message, file=sys.stderr)


@register_model("kivi-llama")
class KIVILM(LM):
    """Run every request independently, in order, without a response cache.

    ``model`` and ``tokenizer`` must be supplied together. This injection path
    permits tiny CPU test models; production loading retains KIVI's CUDA loader.
    Constructor token limits are hard caps, including task-specific generation
    limits. Input overflow is rejected by the shared helper, never truncated.
    """

    def __init__(self, model_name=MODEL_NAME, *, group_size=32, residual_length=128,
                 max_new_tokens=512, max_input_tokens=8192, measure_energy=False,
                 power_profile=None, output_dir=None, model=None, tokenizer=None):
        super().__init__()
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a nonempty string")
        _positive_int("group_size", group_size)
        if group_size not in (8, 16, 32, 64, 128):
            raise ValueError("group_size must be one of 8, 16, 32, 64, 128")
        _positive_int("residual_length", residual_length)
        if residual_length % group_size:
            raise ValueError("residual_length must be divisible by group_size")
        _positive_int("max_new_tokens", max_new_tokens)
        _positive_int("max_input_tokens", max_input_tokens)
        if type(measure_energy) is not bool:
            raise ValueError("measure_energy must be bool")
        if power_profile is not None and not measure_energy:
            raise ValueError("power_profile requires measure_energy=True")
        if (model is None) != (tokenizer is None):
            raise ValueError("supply model and tokenizer together")
        self.model_name = model_name
        self.group_size = group_size
        self.residual_length = residual_length
        self._max_gen_toks = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self.measure_energy = measure_energy
        self.power_profile = power_profile
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.records = []
        self.preloaded = model is not None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if (self.output_dir / "records.jsonl").exists():
                raise FileExistsError(f"records.jsonl already exists in {self.output_dir}")
        self.model, self.tokenizer = model, tokenizer

    def _ensure_loaded(self):
        if self.model is None:
            model, tokenizer = kivi_llama.load_model(
                self.model_name, k_bits=2, v_bits=2, group_size=self.group_size,
                residual_length=self.residual_length,
            )
            expected = {"k_bits": 2, "v_bits": 2, "group_size": self.group_size,
                        "residual_length": self.residual_length}
            for name, value in expected.items():
                actual = getattr(model.config, name, None)
                if type(actual) is not int or actual != value:
                    raise ValueError(f"loaded KIVI config {name} does not match requested {value}")
            if not any(cls.__name__ == "LlamaForCausalLM_KIVI" and
                       cls.__module__ == "kvolt_runtime.backend.vendor.llama_kivi"
                       for cls in type(model).__mro__):
                raise ValueError("loaded model must use the project's LlamaForCausalLM_KIVI backend")
            self.model, self.tokenizer = model, tokenizer
        self._device = getattr(self.model, "device", None)

    @property
    def tokenizer_name(self):
        return self.model_name

    @property
    def batch_size(self):
        return 1

    @property
    def max_length(self):
        return self.max_input_tokens

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    @property
    def eot_token_id(self):
        self._ensure_loaded()
        return self.tokenizer.eos_token_id

    @property
    def device(self):
        self._ensure_loaded()
        return self._device

    def tok_encode(self, string, **kwargs):
        """Expose the actual model tokenizer for task preparation, without truncation."""
        if kwargs.get("truncation") or kwargs.get("left_truncate_len") is not None:
            raise ValueError("input truncation is not supported")
        self._ensure_loaded()
        return self.tokenizer.encode(string, **kwargs)

    def tok_decode(self, tokens, **kwargs):
        self._ensure_loaded()
        return self.tokenizer.decode(tokens, **kwargs)

    def chat_template(self, chat_template=False):
        if chat_template:
            raise ValueError("chat templates are not supported; use plain task prompts")
        return ""

    def loglikelihood(self, requests):
        raise NotImplementedError("KIVILM supports generate_until only")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("KIVILM supports generate_until only")

    def _generation_settings(self, kwargs):
        if not isinstance(kwargs, dict):
            raise TypeError("generation arguments must be a dictionary")
        allowed = {"until", "max_gen_toks", "max_new_tokens", "do_sample",
                   "num_beams", "use_cache", "temperature"}
        unsupported = set(kwargs) - allowed
        if unsupported:
            raise ValueError(f"unsupported generation arguments: {sorted(unsupported)}")
        if "do_sample" in kwargs and kwargs["do_sample"] is not False:
            raise ValueError("only do_sample=False is supported")
        if "use_cache" in kwargs and kwargs["use_cache"] is not True:
            raise ValueError("only use_cache=True (KV cache) is supported")
        beams = kwargs.get("num_beams", 1)
        if type(beams) is not int or beams != 1:
            raise ValueError("only num_beams=1 is supported")
        temperature = kwargs.get("temperature", 0.0)
        if (type(temperature) not in (int, float) or not math.isfinite(temperature)
                or temperature != 0):
            raise ValueError("only temperature=0.0 is supported")
        lengths = {name: _positive_int(name, kwargs[name]) for name in
                   ("max_gen_toks", "max_new_tokens") if name in kwargs}
        if len(lengths) == 2 and len(set(lengths.values())) != 1:
            raise ValueError("max_gen_toks conflicts with max_new_tokens")
        requested = next(iter(lengths.values()), self._max_gen_toks)
        until = kwargs.get("until", ())
        if isinstance(until, str):
            until = (until,)
        elif isinstance(until, list) or until == ():
            until = tuple(until)
        else:
            raise ValueError("until must be a string or list of strings")
        if any(not isinstance(stop, str) or not stop for stop in until):
            raise ValueError("until must contain nonempty strings")
        return {"max_new_tokens": min(requested, self._max_gen_toks), "until": until}

    def _save_energy(self, ordinal, energy, receipt):
        if self.output_dir is None or (energy is None and receipt is None):
            return None
        from kvolt_runtime.telemetry import samples_csv

        relative = f"energy/{ordinal:06d}"
        root = self.output_dir / relative
        root.mkdir(parents=True, exist_ok=True)
        receipt = receipt or {}
        payloads = {
            "energy.json": _dump(energy) + "\n",
            "power-samples.csv": samples_csv(receipt.get("samples", [])),
            "power-trace.jsonl": receipt.get("raw_trace_jsonl", ""),
        }
        for name, payload in payloads.items():
            with (root / name).open("x", encoding="utf-8", newline="") as stream:
                stream.write(payload)
        return relative

    def _record(self, row):
        # Serialize first: invalid JSON must not silently enter in-memory results.
        encoded = _dump(row)
        if self.output_dir is not None:
            mode = "x" if not self.records else "a"
            with (self.output_dir / "records.jsonl").open(mode, encoding="utf-8") as stream:
                stream.write(encoded + "\n")
                stream.flush()
        self.records.append(row)

    def generate_until(self, requests, disable_tqdm=False):
        responses = []
        for request in requests:
            args = request.args
            if not isinstance(args, tuple) or len(args) != 2:
                raise ValueError("generate_until requires (prompt, generation_kwargs)")
            prompt, kwargs = args
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("prompt must be a nonempty string")
            settings = self._generation_settings(kwargs)
            ordinal = len(self.records)
            row = {
                "schema": "kvolt.lm-eval.request.v1", "ordinal": ordinal,
                "task_name": getattr(request, "task_name", None),
                "doc_id": getattr(request, "doc_id", None),
                "repeats": getattr(request, "repeats", None),
                "request_idx": getattr(request, "idx", None),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "model_name": self.model_name, "tokenizer_name": self.tokenizer_name,
                "source": "preloaded_test_model" if self.preloaded else "model.kivi_llama.load_model",
                "model_settings": {"k_bits": 2, "v_bits": 2, "group_size": self.group_size,
                                   "residual_length": self.residual_length,
                                   "max_input_tokens": self.max_input_tokens,
                                   "max_new_tokens_cap": self._max_gen_toks},
                "requested_generation": kwargs.copy(),
                "effective_generation": {**settings, "until": list(settings["until"]),
                                         "do_sample": False, "num_beams": 1, "use_cache": True},
            }
            try:
                self._ensure_loaded()
                layers = getattr(getattr(self.model, "model", None), "layers", ())
                attention = getattr(layers[0], "self_attn", None) if len(layers) else None
                row["actual_backend"] = {
                    "model_class": f"{type(self.model).__module__}.{type(self.model).__name__}",
                    "attention_class": (f"{type(attention).__module__}.{type(attention).__name__}"
                                        if attention is not None else None),
                    "device": str(self._device),
                    "model_revision": getattr(self.model.config, "_commit_hash", None),
                    "model_name_or_path": getattr(self.model.config, "_name_or_path", None),
                    "tokenizer_revision": getattr(self.tokenizer, "init_kwargs", {}).get("_commit_hash"),
                    "model_context_limit": getattr(self.model.config, "max_position_embeddings", None),
                    "kv_config": {name: getattr(self.model.config, name, None) for name in
                                  ("k_bits", "v_bits", "group_size", "residual_length")},
                }
                result = kivi_llama.generate_with_details(
                    self.model, self.tokenizer, prompt, **settings,
                    max_input_tokens=self.max_input_tokens,
                    measure_energy=self.measure_energy, power_profile=self.power_profile,
                )
            except BaseException as original:
                energy = getattr(original, "kvolt_energy_result", None)
                receipt = getattr(original, "kvolt_energy_receipt", None)
                relative = None
                try:
                    relative = self._save_energy(ordinal, energy, receipt)
                except BaseException as error:
                    _diagnostic_error(original, "failed energy artifact save", error)
                try:
                    self._record({**row, "status": "failed", "energy": energy,
                                  "energy_artifacts_dir": relative,
                                  "energy_receipt": self._receipt_metadata(receipt),
                                  "error": {"type": type(original).__name__, "message": str(original)}})
                except BaseException as error:
                    _diagnostic_error(original, "failure record save", error)
                raise
            receipt = result.get("energy_receipt")
            relative = self._save_energy(ordinal, result.get("energy"), receipt)
            self._record({**row, **{key: value for key, value in result.items()
                                   if key != "energy_receipt"},
                          "status": "completed", "energy_artifacts_dir": relative,
                          "energy_receipt": self._receipt_metadata(receipt)})
            responses.append(result["text"])
        return responses

    def _receipt_metadata(self, receipt):
        if receipt is None:
            return None
        if self.output_dir is None:
            # Direct in-memory use has no artifact files to carry the raw trace.
            return receipt
        return {key: value for key, value in receipt.items()
                if key not in ("samples", "raw_trace_jsonl")}
