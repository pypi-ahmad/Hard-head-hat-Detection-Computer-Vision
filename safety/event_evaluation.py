"""One-to-one temporal and spatial matching of reviewed video events."""


def event_scores(predictions, intervals, duration, size):
    if duration <= 0:
        raise ValueError("Positive reviewed video duration required")
    width, height = size
    used, delays = set(), []
    for event in sorted(predictions, key=lambda e: e["confirmed"]):
        x1, y1, x2, y2 = event["box"]
        box = (x1 / width, y1 / height, x2 / width, y2 / height)
        candidates = []
        for index, truth in enumerate(intervals):
            if index in used or not truth["start"] <= event["confirmed"] <= truth["end"]:
                continue
            a, b, c, d = truth["box"]
            intersection = max(0, min(c, box[2])-max(a, box[0])) * max(0, min(d, box[3])-max(b, box[1]))
            union = (c-a)*(d-b)+(box[2]-box[0])*(box[3]-box[1])-intersection
            overlap = intersection / union if union else 0
            if overlap >= .3:
                candidates.append((overlap, index))
        if candidates:
            _, index = max(candidates)
            used.add(index)
            delays.append(event["confirmed"] - intervals[index]["start"])
    return {"reviewed_events": len(intervals), "matched_events": len(used),
            "missed_events": len(intervals)-len(used),
            "event_recall": len(used)/len(intervals) if intervals else None,
            "false_alerts_per_hour": (len(predictions)-len(used))*3600/duration,
            "mean_delay_seconds": sum(delays)/len(delays) if delays else None,
            "matching": "Confirmation within reviewed interval; head-box IoU >= 0.3; one-to-one"}
