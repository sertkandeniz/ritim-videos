# Drop-in replacement for latentsync/utils/face_detector.py (LatentSync commit a229c39).
#
# Upstream uses insightface.app.FaceAnalysis (detection + landmark_2d_106). The InsightFace pretrained
# models are licensed for non-commercial research only, so we must not use them. This module keeps the
# same interface -- FaceDetector(device)(frame) -> (bbox, landmarks[106, 2]) -- but computes the points
# with MediaPipe Face Mesh (Apache-2.0 code and model, bundled in mediapipe==0.10.11 which LatentSync's
# requirements already pin).
#
# LatentSync only reads a few of the 106 points:
#   * image_processor.affine_transform: eyebrow centres [43,48,49,51,50] and 101:106, nose [74,77,83,86]
#     -> the 3-point alignment to its face template (left brow at smaller x, right brow, nose);
#   * this module's own crop: jaw contour + brows + nose bridge (LMK_ADAPT_ORIGIN_ORDER).
# We fill exactly those indices from the corresponding Face Mesh points; the rest stay at the face centre.
# Installed by infra/bootstrap.sh. Verify on the GPU box with:  ritim-latentsync-selftest (see RUNBOOK).

import numpy as np

# MediaPipe Face Mesh indices
_JAW_A = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377]  # image-right side, ear -> chin
_JAW_B = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148]    # image-left side, ear -> chin
_CHIN = 152
_BROW_1 = [70, 63, 105, 66, 107]
_BROW_2 = [300, 293, 334, 296, 336]
_NOSE_BRIDGE = [6, 197]
_NOSE_CENTRE = [197, 195, 5, 4]

LMK_ADAPT_ORIGIN_ORDER = [1, 10, 12, 14, 16, 3, 5, 7, 0, 23, 21, 19, 32, 30, 28, 26, 17,
                          43, 48, 49, 51, 50, 102, 103, 104, 105, 101, 73, 74, 86]


def _interp(points, n):
    points = np.asarray(points, dtype=np.float64)
    src = np.linspace(0, 1, len(points))
    dst = np.linspace(0, 1, n)
    return np.stack([np.interp(dst, src, points[:, 0]), np.interp(dst, src, points[:, 1])], axis=1)


class FaceDetector:
    def __init__(self, device="cuda"):
        import mediapipe as mp

        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5
        )

    def __call__(self, frame, threshold=0.5):
        f_h, f_w, _ = frame.shape
        result = self.mesh.process(np.ascontiguousarray(frame))
        if not result.multi_face_landmarks:
            return None, None
        pts = np.array([[p.x * f_w, p.y * f_h] for p in result.multi_face_landmarks[0].landmark])

        lmk = np.tile(pts.mean(axis=0), (106, 1))
        side_a = _interp(pts[[*_JAW_A, _CHIN]], 17)  # 17 points ear -> chin
        side_b = _interp(pts[[*_JAW_B, _CHIN]], 17)
        lmk[0] = pts[_CHIN]
        lmk[1:17] = side_a[:16]
        lmk[17:33] = side_b[:16]
        brows = sorted([pts[_BROW_1], pts[_BROW_2]], key=lambda b: b[:, 0].mean())
        left, right = brows  # "left" = smaller x, as LatentSync's face template expects
        lmk[[43, 48, 49, 51, 50]] = left
        lmk[101:106] = right
        lmk[73], lmk[74] = pts[_NOSE_BRIDGE[0]], pts[_NOSE_BRIDGE[1]]  # bridge: upstream crop anchor
        lmk[[77, 83, 86]] = pts[_NOSE_CENTRE[1:]]  # with 74 these give the nose centre for alignment
        lmk = np.round(lmk).astype(np.int_)

        # Same crop rule as upstream, on the mapped points.
        halk_face_coord = np.mean([lmk[74], lmk[73]], axis=0)
        sub_lmk = lmk[LMK_ADAPT_ORIGIN_ORDER]
        halk_face_dist = np.max(sub_lmk[:, 1]) - halk_face_coord[1]
        upper_bond = halk_face_coord[1] - halk_face_dist
        x1, y1, x2, y2 = (np.min(sub_lmk[:, 0]), int(upper_bond), np.max(sub_lmk[:, 0]), np.max(sub_lmk[:, 1]))
        w, h = x2 - x1, y2 - y1
        if w < 50 or h < 80 or w / max(h, 1) > 1.5 or w / max(h, 1) < 0.2:
            return None, None
        y2 += int((x2 - x1) * 0.1)
        x1 -= int((x2 - x1) * 0.05)
        x2 += int((x2 - x1) * 0.05)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(f_w, x2), min(f_h, y2)
        return (x1, y1, x2, y2), lmk


def cuda_to_int(cuda_str):
    if cuda_str == "cuda":
        return 0
    return int(str(cuda_str).split(":")[-1])
