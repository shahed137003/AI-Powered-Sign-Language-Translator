\
    import sys
    from pathlib import Path

    # Allow running without installing the package:
    # adds repo root (parent of "scripts/") to sys.path.
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from rendering.cli_render import main


    if __name__ == "__main__":
        main()
