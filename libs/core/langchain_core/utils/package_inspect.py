"""Inspect langchain packages."""

import importlib
import inspect
import logging
import typing
from collections import defaultdict
from enum import Enum
from importlib.metadata import metadata
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence

import toml

from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables.configurable import StrEnum

logger = logging.getLogger(__name__)


class MemberType(StrEnum):
    """The type of the module member."""

    class_ = "class"
    function = "function"


ClassKind = Literal["TypedDict", "Regular", "Pydantic", "enum"]


class MemberInfo(BaseModel):
    """Information about a member of the module."""

    name: str
    """The name of the class."""
    qualified_name: str
    """The fully qualified name of the class."""
    is_public: bool
    """Whether the member is public or not."""
    member_type: Optional[MemberType] = None
    """The type of the member."""
    class_kind: Optional[ClassKind] = None
    """The kind of the class. None for functions."""


class Module(BaseModel):
    """Module with a namespace and members."""

    name: str
    namespace: str
    members: Sequence[MemberInfo]


def get_packages(repo_dir: str | Path, partner_packages: bool) -> Dict[str, Path]:
    """Return package names and paths to the package repo.

    A package can be in the langchain monorepo or in an external repo.
    If the package is in the external repo, the package should be pip-installed
    and the path to the package is inside the 'site-packages/' folder.
    A partner package should have a subfolder with README.md
    in the 'libs/partners/`, even if the package itself is placed in
    the external repo.
    """
    folder_path = (
        (Path(repo_dir) / "libs" / "partners")
        if partner_packages
        else (Path(repo_dir) / "libs")
    )
    package2path = {
        (
            "langchain"
            if d.name == "langchain"
            else ("langchain-" + d.name.replace("_", "-"))
        ): d
        for d in folder_path.iterdir()
        if d.is_dir() and d.name not in ["partners"]
    }
    new_package2path = {}
    for package_name, package_path in package2path.items():
        if is_external_repo(package_path):
            try:
                module = importlib.import_module(package_name.replace("-", "_"))
            except (ModuleNotFoundError, ImportError, TypeError) as e:
                logger.error(
                    f"Error: Unable to import package '{package_name}' with error: {e} "
                    f"Use 'pip install {package_name} -U' to install the package."
                )
            else:
                if isinstance(module.__path__, list) and module.__path__:
                    new_package2path[package_name] = Path(module.__path__[0])
                else:
                    logger.error(
                        f"Error: Unable to find the path to the package '{package_name}'."
                    )
        else:
            new_package2path[package_name] = package_path
    return new_package2path


def is_external_repo(package_path: Path) -> bool:
    """Return True if the package is in the external repo."""
    return not (package_path / "pyproject.toml").exists()


class Package:
    def __init__(
        self, name: str, path: Path, is_partner: bool = False, upload: bool = True
    ):
        """Initialize a package.

        Args:
            name: The name of the package.
            path: The path to the package.
            is_partner: Whether the package is a partner package or not.
            upload: Whether to upload the package parameters or not. Used for testing.
        """
        self.name = name
        self.path = path
        self.is_partner = is_partner
        self.namespace = self.get_namespace(name)
        if upload:
            self.external_repo = is_external_repo(self.path)
            self.version = self.get_version(self.path, package_name=name)
            self.source_path = self.get_source_path(self.path)
            self.modules = self.get_modules(self.path)

    def get_namespace(self, package_name: str) -> str:
        return package_name.replace("-", "_")

    def get_source_path(self, package_path: Path) -> Path | None:
        """Return the path to the directory containing the package source code."""
        if is_external_repo(package_path):
            # if the package is in an external repo, we can't determine the source path
            return None
        source_dir = package_path / self.name.replace("-", "_")
        if not source_dir.exists():
            raise ValueError(f"Source directory {source_dir} does not exist.")
        return source_dir

    def get_version(self, package_path: Path, package_name: str) -> str:
        """Return the version of the package."""
        if is_external_repo(package_path):
            md = metadata(package_name)
            if "Version" in md:
                return md["Version"]
            elif "version" in md:
                return md["version"]
            else:
                raise ValueError(
                    f"Error: Unable to find the version of the package '{package_name}'."
                )
        else:
            try:
                with open(package_path / "pyproject.toml", "r") as f:
                    pyproject = toml.load(f)
            except FileNotFoundError:
                raise ValueError(
                    f"pyproject.toml not found in {package_path} folder.\n"
                )
            return pyproject["tool"]["poetry"]["version"]

    def get_modules(self, package_path: Path) -> Dict[str, Module]:
        """Recursively load modules of a package based on the file system.

        Traversal based on the file system makes it easy to determine which
        of the modules/packages are part of the package vs. 3rd party or built-in.

        Parameters:
            package_path: Path to the package directory.

        Returns:
            A dict of loaded modules.
        """
        qualified_module_name2module = {}
        # Traverse the package directory and load all modules
        for file_path in package_path.rglob("*.py"):
            qualified_module_name = file_path.relative_to(package_path)
            qualified_module_name = (
                str(qualified_module_name).replace(".py", "").replace("/", ".")
            )
            # Skip the modules that are not part of the package, like "tests".
            if qualified_module_name.split(".")[0] != self.namespace:
                continue

            try:
                qualified_module_name2module[qualified_module_name] = Module(
                    name=qualified_module_name.split(".")[-1],
                    namespace=qualified_module_name,
                    members=load_module_members(
                        qualified_module_name=qualified_module_name
                    ),
                )
            except (ImportError, AttributeError, TypeError) as e:
                logger.error(
                    f"Error: Unable to import module '{qualified_module_name}' "
                    f"with error: {e}"
                )  # noqa: E501

        return qualified_module_name2module


class Repo:
    """A repository with packages. TODO"""

    def __init__(self, dir: str):
        self.dir = dir
        self.packages = get_packages(dir, False)
        self.partner_packages = get_packages(dir, True)


def load_module_members(qualified_module_name: str) -> List[MemberInfo]:
    """Load all members of a module.

    Args:
        qualified_module_name: full absolute name of the module.
          It does not process the relative imports.

    Returns:
        list: A list of loaded module members.
    """
    members: List[MemberInfo] = []
    module = importlib.import_module(qualified_module_name)
    for name, type_ in inspect.getmembers(module):
        if not hasattr(type_, "__module__"):
            continue
        if type_.__module__ != qualified_module_name:
            continue
        member = MemberInfo(
            name=name,
            qualified_name=f"{qualified_module_name}.{name}",
            is_public=not name.startswith("_"),
        )
        if inspect.isclass(type_):
            member.member_type = MemberType.class_
            if isinstance(type_, typing._TypedDictMeta):  # type: ignore
                member.class_kind = "TypedDict"
            elif issubclass(type_, Enum):
                member.class_kind = "enum"
            elif issubclass(type_, BaseModel):
                member.class_kind = "Pydantic"
            else:
                member.class_kind = "Regular"
        elif inspect.isfunction(type_):
            member.member_type = MemberType.function
        else:
            logger.warning(
                f"{qualified_module_name}: {name=}: Unknown member type: {type_=}"
            )
            continue
        members.append(member)
    return members


def top_two_levels_of_modules(
    package: Package, remove_hidden: bool = False, remove_empty: bool = True
) -> Dict[str, List[MemberInfo]]:
    """Return the top two levels of modules.

    Removes the namespaces without members.
    It can remove 'hidden' namespaces that have parts that started
    with `_`. For example, `langchain_core._api`.

    Args:
        package: The package to inspect.
        remove_hidden: Whether to remove hidden namespaces or not.
        remove_empty: Whether to remove namespaces without members or not.

    Returns:

    """

    def _has_hidden_part(namespace: str) -> bool:
        return any(part.startswith("_") for part in namespace.split("."))

    _top_two_levels2module_members: defaultdict[str, list] = defaultdict(list)
    for qualified_module_name, module in package.modules.items():
        namespace_parts = qualified_module_name.split(".")
        top_two_levels = ".".join(namespace_parts[:2])
        _top_two_levels2module_members[top_two_levels] += module.members
    top_two_levels2module_members: dict[str, list] = dict(
        _top_two_levels2module_members
    )
    if remove_empty:
        top_two_levels2module_members = {
            k: v for k, v in top_two_levels2module_members.items() if v
        }
    if remove_hidden:
        top_two_levels2module_members = {
            k: v
            for k, v in top_two_levels2module_members.items()
            if not _has_hidden_part(k)
        }
    # sort by keys:
    top_two_levels2module_members = {
        k: v for k, v in sorted(top_two_levels2module_members.items())
    }
    return top_two_levels2module_members
