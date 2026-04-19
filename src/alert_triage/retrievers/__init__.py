"""Retriever contracts and phase-0 reference implementations."""

from ._maxsim_hamming_ref import maxsim_hamming
from ._maxsim_ref import maxsim_cosine
from .base import Candidate, QueryBundle, Retriever, TokenVectorStore
from .binary_rerank import BinaryThenFP16RerankRetriever
from .hamming_udf import HammingUDFRetriever

__all__ = [
    "BinaryThenFP16RerankRetriever",
    "Candidate",
    "HammingUDFRetriever",
    "QueryBundle",
    "Retriever",
    "TokenVectorStore",
    "maxsim_cosine",
    "maxsim_hamming",
]
