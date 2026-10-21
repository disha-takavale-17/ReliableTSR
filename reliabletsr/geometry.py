"""All boxes use image pixel coordinates [left, top, right, bottom]."""

def area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection(a, b):
    return [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]


def iou(a, b):
    common = area(intersection(a, b))
    union = area(a) + area(b) - common
    return common / union if union else 0.0


def coverage(a, b):
    """Fraction of a covered by b."""
    return area(intersection(a, b)) / area(a) if area(a) else 0.0


def union_box(boxes):
    if not boxes:
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]
