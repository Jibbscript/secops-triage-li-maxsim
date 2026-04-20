"""Retriever contracts and phase-0 reference implementations."""

from ._maxsim_hamming_ref import maxsim_hamming
from ._maxsim_ref import maxsim_cosine
from .base import Candidate, FilterClause, QueryBundle, QueryFilter, Retriever, TokenVectorStore
from .bm25 import BM25Retriever
from .binary_rerank import BinaryThenFP16RerankRetriever
from .fp16_ref import FP16ReferenceRetriever
from .hamming_udf import HammingUDFRetriever
from .hybrid_rerank import HybridBM25ThenFP16RerankRetriever
from .lance_mv import LanceMVRetriever

__all__ = [
    "BM25Retriever",
    "BinaryThenFP16RerankRetriever",
    "Candidate",
    "FilterClause",
    "FP16ReferenceRetriever",
    "HammingUDFRetriever",
    "HybridBM25ThenFP16RerankRetriever",
    "LanceMVRetriever",
    "QueryBundle",
    "QueryFilter",
    "Retriever",
    "TokenVectorStore",
    "maxsim_cosine",
    "maxsim_hamming",
]
