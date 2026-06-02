# verl/utils/dataproto_utils.py
from __future__ import annotations
from typing import Iterable

def normalize_non_tensor_batch(dp, to_meta: Iterable[str] = ()):
    """
    Fix dp.non_tensor_batch so DataProto.chunk can split it safely:
    - Move known per-batch constants to meta_info.
    - Broadcast any scalar/0-D values to length-B lists.
    """
    if dp is None or not hasattr(dp, "batch"):
        return dp
    # infer B
    try:
        B = next(iter(dp.batch.values())).shape[0]
    except Exception:
        return dp

    nt = dp.non_tensor_batch
    if nt is None:
        return dp

    # 1) move known constants to meta_info
    for k in list(nt.keys()):
        if k in to_meta:
            dp.meta_info[k] = nt[k]
            del nt[k]

    # 2) broadcast anything not length-B
    for k, v in list(nt.items()):
        try:
            if hasattr(v, "__len__") and not isinstance(v, (str, bytes)) and len(v) == B:
                continue  # already per-sample
        except Exception:
            pass
        nt[k] = [v] * B
    return dp