import os


class PathResolver:
    def __init__(self, config):
        self.config = config
        self.project_root = self._find_project_root()

    def resolve_path(self, relative_path):
        """Convert a relative path to an absolute path from project root"""
        return os.path.join(self.project_root, relative_path)

    def _find_project_root(self):
        """Find the project root directory"""
        return os.getcwd()
