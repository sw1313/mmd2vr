"""Make the diffusion reference and final panorama from the same measured stage atlas."""

import math

import cv2
import numpy as np


def project(atlas, mask, source_width, angle, roll, size, world=None):
    """Project measured pixels to an ERP in strips, retaining their native detail."""
    ah, aw = mask.shape
    f = source_width * 0.5 / math.tan(math.radians(angle * 0.5))
    lon = ((np.arange(2 * size, dtype="f") + 0.5) / (2 * size) - 0.5) * 2 * np.pi
    dx = f * np.tan(lon)
    front = np.abs(lon) < np.pi / 2
    cr, sr = math.cos(roll), math.sin(roll)
    if world is None:
        result = np.empty((size, 2 * size, 3), np.uint8)
        result[:] = (0, 255, 0)
    else:
        result = world
    for start in range(0, size, 96):
        slab = slice(start, min(start + 96, size))
        lat = ((np.arange(slab.start, slab.stop, dtype="f") + 0.5) / size - 0.5) * np.pi
        dy = f * np.tan(lat[:, None]) / np.cos(lon)
        x = (aw / 2 - 0.5 + cr * dx - sr * dy).clip(-2, aw + 1).astype("f")
        y = (ah / 2 - 0.5 + sr * dx + cr * dy).clip(-2, ah + 1).astype("f")
        measured = cv2.remap(atlas, x, y, cv2.INTER_LINEAR)
        support = cv2.remap(mask, x, y, cv2.INTER_LINEAR)
        support[:, ~front] = 0
        if world is None:
            valid = support > 120
            result[slab][valid] = measured[valid]
        else:
            weight = support[:, :, None].astype("f") / 255
            result[slab] = np.rint(
                result[slab] * (1 - weight) + measured * weight
            ).astype("uint8")
    return result


def bake(world, atlas, mask, width, angle, roll):
    distance = cv2.distanceTransform((mask > 45).astype("uint8"), cv2.DIST_L2, 3)
    confidence = np.minimum(
        mask, 255 * np.clip(distance / max(1, width * 0.045), 0, 1)
    ).astype("uint8")
    return project(atlas, confidence, width, angle, roll, world.shape[0], world)


def foreground_predictor(weights):
    from segment_anything import SamPredictor, sam_model_registry

    return SamPredictor(
        sam_model_registry["vit_b"](checkpoint=str(weights)).eval().to("cuda")
    )


def scenery_mask(image, predictor):
    h, w = image.shape[:2]
    # Segment the central performer with the same conservative prompts as the reference workflow.
    predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    masks, scores, _ = predictor.predict(
        point_coords=np.float32([[0.5 * w, 0.41 * h], [0.48 * w, 0.6 * h]]),
        point_labels=np.ones(2, dtype="int32"),
        box=np.float32([0.17 * w, 0, 0.84 * w, h - 1]),
        multimask_output=True,
    )
    good = masks[
        (scores > 0.72)
        & (masks.mean(axis=(1, 2)) > 0.08)
        & (masks.mean(axis=(1, 2)) < 0.85)
    ]
    actor = np.any(good, axis=0) if len(good) else np.zeros((h, w), bool)
    if not len(good):
        actor[:, int(0.23 * w) : int(0.77 * w)] = True
    actor[int(0.1 * h) :, int(0.34 * w) : int(0.66 * w)] = True
    k = max(3, 2 * round(w * 0.009) + 1)
    actor = cv2.dilate(
        actor.astype("uint8"), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )
    mask = 1 - actor
    pad = max(2, round(0.004 * w))
    mask[:pad] = mask[-pad:] = 0
    mask[:, :pad] = mask[:, -pad:] = 0
    return np.minimum(
        cv2.distanceTransform(mask, cv2.DIST_L2, 3) / max(3, w * 0.012), 1
    )
