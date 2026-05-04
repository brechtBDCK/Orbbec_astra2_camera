import cv2
import numpy as np


def make_preview(color_image, depth_image, display_width: int, display_height: int):
    if color_image is None and depth_image is None:
        return None
    if color_image is None:
        return cv2.resize(depth_image, (display_width, display_height), interpolation=cv2.INTER_AREA)
    if depth_image is None:
        return cv2.resize(color_image, (display_width, display_height), interpolation=cv2.INTER_AREA)

    color_image = cv2.resize(
        color_image,
        (display_width // 2, display_height),
        interpolation=cv2.INTER_AREA,
    )
    depth_image = cv2.resize(
        depth_image,
        (display_width // 2, display_height),
        interpolation=cv2.INTER_AREA,
    )
    return np.hstack((color_image, depth_image))


def draw_info_panel(image,color_profile_text: str,depth_profile_text: str,filters: list[str] | None = None):
    if image is None:
        return None

    filters = filters or []
    lines = [
        "Color: " + color_profile_text,
        "Depth: " + depth_profile_text,
        "Filters:",
        *filters,
    ]

    panel_width = 420
    line_height = 22
    text_top = 26
    panel_height = max(150, text_top + len(lines) * line_height + 12)
    padding = 16
    x0 = max(0, image.shape[1] - panel_width - padding)
    y0 = padding
    x1 = min(image.shape[1], x0 + panel_width)
    y1 = min(image.shape[0], y0 + panel_height)

    cv2.rectangle(image, (x0, y0), (x1, y1), (255, 255, 255), thickness=-1)
    cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 0), thickness=1)

    text_x = x0 + 12
    text_y = y0 + text_top
    for line in lines:
        cv2.putText( image, line, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 1, cv2.LINE_AA)
        text_y += line_height

    return image
