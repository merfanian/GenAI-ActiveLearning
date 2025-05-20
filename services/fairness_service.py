from .model_service import predict

def calculate_group_performances(model_path: str, metadata_df, all_image_paths: list[str]) -> dict:
    preds = predict(model_path, all_image_paths)
    df = metadata_df.copy()
    df["predicted_label"] = [p["predicted_label"] for p in preds]
    attrs = [c for c in df.columns if c not in ("filename", "label", "predicted_label")]
    group_perfs = {}
    if attrs:
        groups = df.groupby(attrs)
    else:
        groups = [((), df)]
    for group_vals, group_df in groups:
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        key = tuple(zip(attrs, group_vals))
        correct = (group_df["predicted_label"] == group_df["label"]).sum()
        count = len(group_df)
        acc = correct / count if count else 0.0
        group_perfs[key] = {"accuracy": acc, "count": count}
    return group_perfs

def find_worst_performing_group(group_performances: dict) -> dict:
    if not group_performances:
        return {}
    worst = min(group_performances.items(), key=lambda x: x[1]["accuracy"])
    attrs_tuple, metrics = worst
    return {"attributes": dict(attrs_tuple), "accuracy": metrics["accuracy"], "count": metrics["count"]}