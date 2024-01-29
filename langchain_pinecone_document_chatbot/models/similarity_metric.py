from enum import Enum


class SimilarityMetric(str, Enum):  # noqa: WPS600
    DOT_PRODUCT = "dotproduct"
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
