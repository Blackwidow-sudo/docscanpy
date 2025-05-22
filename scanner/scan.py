import cv2
import numpy as np


class Scanner:
    @staticmethod
    def find_contour(img: np.ndarray) -> np.ndarray:
        # Preprocess
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0, 0, cv2.BORDER_DEFAULT)
        _, thresh = cv2.threshold(
            blur,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in contours:
            epsilon = 0.1 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) == 4:
                return approx
        return None

    @staticmethod
    def order_points(pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    @staticmethod
    def draw_contour(img: np.ndarray, pts: np.ndarray):
        cpy = img.copy()
        cv2.drawContours(cpy, [pts], 0, (0, 255, 0), 5)
        return cpy

    @staticmethod
    def transform_perspective(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
        ordered_pts = Scanner.order_points(pts.reshape(4, 2))
        (tl, tr, br, bl) = ordered_pts
        width = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
        height = max(int(np.linalg.norm(tr - br)),
                     int(np.linalg.norm(tl - bl)))
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
        M = cv2.getPerspectiveTransform(ordered_pts, dst)
        warped = cv2.warpPerspective(img, M, (width, height))
        return warped

    @staticmethod
    def document_filter(img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bil = cv2.bilateralFilter(gray, 9, 75, 75)
        return cv2.adaptiveThreshold(
            bil,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
