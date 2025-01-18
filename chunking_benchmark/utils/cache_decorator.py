import os
import pickle
import hashlib
import inspect
import functools
import json


def cache_to_file(location, verbose=False):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            args_dict = bound_args.arguments

            cache_args = {}
            for key in ["question", "index_name", "namespace", "retriever_topk", "rerank_topk", "filter", "to_rerank"]:
                value = args_dict.get(key)
                if key == "filter":
                    value = json.dumps(value, sort_keys=True)
                cache_args[key] = value

            key_items = [f"{k}={v}" for k, v in cache_args.items()]
            key_str = "_".join(key_items)
            key_hash = hashlib.md5(key_str.encode("utf-8")).hexdigest()
            filename = os.path.join(
                location,
                f"torerank_{cache_args.get("to_rerank")}_{cache_args.get("index_name","")}_{cache_args.get("namespace","")}_{args_dict.get("filter",{}).get("chapter_name")}_{key_hash}.json",
            )
            metadata_filename = filename.replace(".json", ".metadata.json")

            if os.path.exists(filename):
                if verbose:
                    print(f"Cache hit: {cache_args}")
                with open(filename, "r", encoding="utf-8") as f:
                    result = json.load(f)
                return result

            result = func(*args, **kwargs)
            os.makedirs(location, exist_ok=True)
            if verbose:
                print(f"Caching result {cache_args}")
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(result, f)

            with open(metadata_filename, "w", encoding="utf-8") as f:
                cache_args_serializable = cache_args.copy()
                if "filter" in cache_args_serializable:
                    cache_args_serializable["filter"] = json.loads(cache_args_serializable["filter"])
                json.dump(cache_args_serializable, f, indent=2)
            return result

        return wrapper

    return decorator
