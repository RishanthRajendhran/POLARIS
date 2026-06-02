"""
Download LoRA adapter from Tinker cloud and merge with base model → HF checkpoint.

Usage:
    python3 scripts/export_sft_checkpoint.py \
        --tinker-weights-path "tinker://YOUR_RUN/sampler_weights/step_XXXXXX" \
        --base-model "Qwen/Qwen3-8B" \
        --out-dir ./outputs/merged_hf
"""
import argparse
import os
import tarfile
import tempfile
import urllib.request
from pathlib import Path


def _safe_extract_tar(tar: tarfile.TarFile, path: Path) -> None:
    base = str(path.resolve())
    for m in tar.getmembers():
        dest = str((path / m.name).resolve())
        if not dest.startswith(base):
            raise RuntimeError(f"Unsafe tar member path: {m.name}")
    tar.extractall(path=path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tinker-weights-path", required=True,
                        help="tinker:// path from checkpoints.jsonl (tinker_weights_path field)")
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B",
                        help="Base model name or path")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for merged HF model")
    parser.add_argument("--hf-cache", default=os.getenv("HF_HOME", "./.cache/huggingface"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[export] Downloading adapter from {args.tinker_weights_path} ...")
    import asyncio
    import tinker
    service = tinker.ServiceClient()
    rest = service.create_rest_client()

    async def _get_url():
        result = rest.get_checkpoint_archive_url_from_tinker_path_async(args.tinker_weights_path)
        if asyncio.isfuture(result) or asyncio.iscoroutine(result):
            return await result
        # may be a concurrent.futures.Future
        if hasattr(result, "result"):
            return result.result()
        return result

    url_resp = asyncio.run(_get_url())

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        tar_path = tmp / "adapter.tar"
        print(f"[export] Fetching tar ...")
        urllib.request.urlretrieve(url_resp.url, str(tar_path))

        adapter_dir = tmp / "adapter"
        adapter_dir.mkdir()
        with tarfile.open(tar_path, "r") as tar:
            _safe_extract_tar(tar, adapter_dir)
        tar_path.unlink(missing_ok=True)
        print(f"[export] Adapter extracted to {adapter_dir}")
        print(f"[export] Adapter contents: {list(adapter_dir.iterdir())}")

        print(f"[export] Loading base model {args.base_model} ...")
        os.environ.setdefault("HF_HOME", args.hf_cache)

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype="auto",
            trust_remote_code=True,
            cache_dir=args.hf_cache,
        )
        print(f"[export] Loading LoRA adapter ...")
        peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
        print(f"[export] Merging LoRA weights ...")
        merged = peft_model.merge_and_unload()

        print(f"[export] Saving merged model to {out_dir} ...")
        merged.save_pretrained(str(out_dir))

        tok = AutoTokenizer.from_pretrained(
            args.base_model, trust_remote_code=True, cache_dir=args.hf_cache
        )
        tok.save_pretrained(str(out_dir))

    print(f"[export] Done. Merged HF model at: {out_dir}")


if __name__ == "__main__":
    main()
