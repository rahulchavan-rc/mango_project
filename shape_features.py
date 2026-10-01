from __future__ import annotations

import tensorflow as tf


SHAPE_FEATURE_COUNT = 8


def extract_shape_features(image: tf.Tensor) -> tf.Tensor:
    """Measure coarse mango silhouette geometry from a resized RGB image."""
    image = tf.cast(image, tf.float32)
    border = tf.concat(
        [
            image[0, :, :],
            image[-1, :, :],
            image[:, 0, :],
            image[:, -1, :],
        ],
        axis=0,
    )
    background = tf.reduce_mean(border, axis=0)
    foreground = tf.norm(image - background, axis=-1) > 25.0
    coordinates = tf.where(foreground)

    def measure() -> tf.Tensor:
        rows = coordinates[:, 0]
        columns = coordinates[:, 1]
        min_row = tf.reduce_min(rows)
        max_row = tf.reduce_max(rows)
        min_column = tf.reduce_min(columns)
        max_column = tf.reduce_max(columns)
        height = tf.cast(max_row - min_row + 1, tf.float32)
        width = tf.cast(max_column - min_column + 1, tf.float32)
        area = tf.cast(tf.size(coordinates), tf.float32)
        image_height = tf.cast(tf.shape(image)[0], tf.float32)
        image_width = tf.cast(tf.shape(image)[1], tf.float32)
        area_ratio = area / (image_height * image_width)
        aspect_ratio = width / height
        fill_ratio = area / (height * width)
        centroid_x = (tf.reduce_mean(tf.cast(columns, tf.float32)) - tf.cast(min_column, tf.float32)) / width
        centroid_y = (tf.reduce_mean(tf.cast(rows, tf.float32)) - tf.cast(min_row, tf.float32)) / height

        row_widths = tf.reduce_sum(
            tf.cast(foreground[min_row : max_row + 1, min_column : max_column + 1], tf.float32),
            axis=1,
        )
        row_count = tf.shape(row_widths)[0]
        sample_rows = tf.cast(
            tf.round(tf.cast(row_count - 1, tf.float32) * tf.constant([0.25, 0.5, 0.75])),
            tf.int32,
        )
        profile_widths = tf.gather(row_widths, sample_rows) / width
        return tf.concat(
            [
                tf.stack([area_ratio, aspect_ratio, fill_ratio, centroid_x, centroid_y]),
                profile_widths,
            ],
            axis=0,
        )

    return tf.cond(
        tf.shape(coordinates)[0] > 0,
        measure,
        lambda: tf.zeros([SHAPE_FEATURE_COUNT], dtype=tf.float32),
    )
