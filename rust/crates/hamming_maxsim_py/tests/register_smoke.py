from __future__ import annotations

from datafusion import SessionContext

from hamming_maxsim_py import register


def main() -> None:
    ctx = SessionContext()
    register(ctx)


if __name__ == "__main__":
    main()
