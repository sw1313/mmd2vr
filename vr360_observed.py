import numpy as np, cv2 as cv

W, H, pad = 480, 270, 44


def orient(im, side):
    if side == 0:
        return im
    if side == 1:
        return im[:, ::-1]
    trans = im.swapaxes(0, 1)
    return trans if side == 2 else trans[:, ::-1]


def align_one(
    target, donors, target_index, donor_indices, gray_target=None, gray_donors=None
):
    canvas = np.zeros((H + 2 * pad, W + 2 * pad, 3), np.uint8)
    valid = np.zeros(canvas.shape[:2], np.float32)
    if gray_target is None:
        gray_target = cv.cvtColor(target, cv.COLOR_BGR2GRAY)
    if gray_donors is None:
        gray_donors = [cv.cvtColor(image, cv.COLOR_BGR2GRAY) for image in donors]
    for side in range(4):
        src = orient(target, side)
        ref = [orient(image, side) for image in donors]
        gray = np.ascontiguousarray(orient(gray_target, side))
        grayref = [orient(image, side) for image in gray_donors]
        out = orient(canvas, side)
        cons = orient(valid, side)
        h, w = gray.shape
        for v0 in range(0, h - 23, 16):
            n = min(32, h - v0)
            # Leave the five real pixels nearest the boundary out of matching.
            crop = gray[v0 : v0 + n, 5:27]
            if crop.std() < 4:
                continue
            candidates = []
            for j, d in enumerate(grayref):
                if abs(target_index - donor_indices[j]) < 4:
                    continue
                lo = max(v0 - 42, 0)
                hi = min(v0 + n + 42, h)
                s = d[lo:hi, :111]
                scores = cv.matchTemplate(s, crop, cv.TM_CCOEFF_NORMED)
                _, score, _, (x, y) = cv.minMaxLoc(scores)
                dx = x - 5
                dy = lo + y - v0
                if dx < 3 or score < 0.88:
                    continue
                l = v0 + dy
                test = ref[j][l : l + n, dx : dx + 5]
                err = np.abs(test.astype("f") - src[v0 : v0 + n, :5]).mean()
                if err > 9:
                    continue
                candidates.append((score - err * 0.008, j, l, dx))
            candidates.sort(reverse=True)
            if len(candidates) < 2:
                continue
            # Require independent observations, aligned on the hidden strip,
            # and agreement at the actual outside pixels before accepting them.
            best = np.zeros((n, pad), np.float32)
            color = np.zeros((n, pad, 3), np.uint8)
            for i, (s, j, l, dx) in enumerate(candidates[:5]):
                for t, k, ll, dd in candidates[i + 1 : 9]:
                    if abs(donor_indices[j] - donor_indices[k]) < 5:
                        continue
                    extent = min(pad, dx - 1, dd - 1)
                    if extent < 2:
                        continue
                    a = ref[j][l : l + n, dx - extent : dx]
                    b = ref[k][ll : ll + n, dd - extent : dd]
                    diff = np.abs(a.astype("f") - b).mean(2)
                    residual = cv.blur(diff, (3, 3))
                    agree = np.minimum(
                        np.clip((12 - residual) / 7, 0, 1),
                        np.clip((23 - diff) / 9, 0, 1),
                    )
                    # Remove islands beyond a disagreement along the same material.
                    confidence = np.minimum.accumulate(agree[:, ::-1], axis=1)[
                        :, ::-1
                    ] * min(s, t)
                    sl = best[:, -extent:]
                    good = confidence > sl
                    sl[good] = confidence[good]
                    color[:, -extent:][good] = (
                        (a.astype("uint16") + b + 1) // 2
                    ).astype("uint8")[good]
            view = cons[pad + v0 : pad + v0 + n, :pad]
            take = best > view
            out[pad + v0 : pad + v0 + n, :pad][take] = color[take]
            view[take] = best[take]
    return canvas, np.uint8(np.clip(valid * 255, 0, 255))
