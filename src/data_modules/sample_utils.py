"""
Small helpers to let the active-learning code work with both dataset styles:
    1) tuple/list samples: (image, target)
    2) dict samples: {"image": image, "labels": target, ...}
"""


def unpack_sample(sample):
    """Return (image, target) from either a dict sample or tuple/list sample."""
    if isinstance(sample, dict):
        image = sample.get("image", sample.get("img"))
        target = sample.get("labels", sample.get("label", sample.get("target")))
        if image is None:
            raise KeyError("Dictionary sample does not contain an 'image' key.")
        return image, target

    if isinstance(sample, (tuple, list)) and len(sample) >= 2:
        return sample[0], sample[1]

    raise TypeError(f"Unsupported sample type: {type(sample)}")


def unpack_batch(batch):
    """Return (images, targets) from either a dict batch or tuple/list batch."""
    if isinstance(batch, dict):
        images = batch.get("image", batch.get("images"))
        targets = batch.get("labels", batch.get("label", batch.get("target")))
        if images is None:
            raise KeyError("Dictionary batch does not contain an 'image' key.")
        return images, targets

    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]

    raise TypeError(f"Unsupported batch type: {type(batch)}")
