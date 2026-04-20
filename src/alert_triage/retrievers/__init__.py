"""Retriever contracts and phase-0 reference implementations."""

from ._maxsim_hamming_ref import maxsim_hamming
from ._maxsim_ref import maxsim_cosine
from .base import Candidate, FilterClause, QueryBundle, QueryFilter, Retriever, TokenVectorStore
from .binary_rerank import BinaryThenFP16RerankRetriever
from .fp16_ref import FP16ReferenceRetriever
from .hamming_udf import HammingUDFRetriever

__all__ = [
    "BinaryThenFP16RerankRetriever",
    "Candidate",
    "FilterClause",
    "FP16ReferenceRetriever",
    "HammingUDFRetriever",
    "QueryBundle",
    "QueryFilter",
    "Retriever",
    "TokenVectorStore",
    "maxsim_cosine",
    "maxsim_hamming",
]
