from pathlib import Path


class PathPolicyError(Exception):
    pass


class PathPolicy:
    def __init__(self, allowed_paths: tuple[Path, ...]) -> None:
        self._allowed_paths = tuple(
            path.resolve()
            for path in allowed_paths
        )


    def resolve(self, path: Path) -> Path:
        resolved = path.resolve()

        if not self._is_allowed(resolved):
            raise PathPolicyError(
                f"Path '{path}' is outside allowed paths"
            )

        return resolved
    

    def _is_allowed(self, path: Path) -> bool:
        return any(
            path == allowed
            or allowed in path.parents
            for allowed in self._allowed_paths
        )