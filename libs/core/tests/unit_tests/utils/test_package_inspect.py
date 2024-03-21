from pathlib import Path

import pytest

from langchain_core.utils.package_inspect import (
    MemberInfo,
    MemberType,
    Module,
    Package,
    get_packages,
    is_external_repo,
    load_module_members,
    top_two_levels_of_modules,
)

# dir of the mono-repo root if the current file is
# in langchain/libs/core/tests/unit_tests/utils
ROOT_DIR = Path(__file__).parents[5]


def test_get_packages_partners() -> None:
    ret = get_packages(ROOT_DIR, partner_packages=True)
    assert {
        "langchain-elasticsearch",
        "langchain-exa",
        "langchain-mongodb",
        "langchain-together",
        "langchain-groq",
        "langchain-robocorp",
        "langchain-openai",
        "langchain-anthropic",
        "langchain-airbyte",
        "langchain-pinecone",
        "langchain-nomic",
        "langchain-mistralai",
        "langchain-fireworks",
        "langchain-ibm",
        "langchain-ai21",
    }.issubset(set(ret))
    for path in ret.values():
        assert path.exists()


def test_get_packages_non_partner() -> None:
    ret = get_packages(ROOT_DIR, partner_packages=False)
    assert set(ret) == {
        "langchain",
        "langchain-cli",
        "langchain-community",
        "langchain-core",
        "langchain-experimental",
        "langchain-text-splitters",
    }
    for path in ret.values():
        assert path.exists()


@pytest.mark.parametrize(
    "package_name",
    [
        "langchain",
        "langchain-cli",
        "langchain-community",
        "langchain-core",
        "langchain-experimental",
        "langchain-text-splitters",
    ],
)
def test_package_init(package_name: str) -> None:
    main_packages = get_packages(ROOT_DIR, partner_packages=False)
    package = Package(
        package_name, main_packages[package_name], is_partner=False, upload=False
    )
    assert package.name == package_name
    assert package.namespace == package_name.replace("-", "_")
    libs_dir = (
        package_name[len("langchain-") :]
        if package_name != "langchain"
        else "langchain"
    )
    assert package.path == Path(ROOT_DIR / "libs" / libs_dir)
    assert not package.is_partner


@pytest.mark.parametrize(
    "package_name",
    [
        "langchain-ai21",
        "langchain-airbyte",
        "langchain-anthropic",
        "langchain-elasticsearch",
    ],
)
def test_package_init_partner(package_name: str) -> None:
    partner_packages = get_packages(ROOT_DIR, partner_packages=True)
    package = Package(
        package_name, partner_packages[package_name], is_partner=True, upload=False
    )
    assert package.name == package_name
    assert package.namespace == package_name.replace("-", "_")
    libs_dir = (
        package_name[len("langchain-") :]
        if package_name != "langchain"
        else "langchain"
    )
    assert package.path == Path(ROOT_DIR / "libs" / "partners" / libs_dir)
    assert package.is_partner


def test_get_package_version() -> None:
    for is_partner_packages in [False, True]:
        for package_name, package_path in get_packages(
            ROOT_DIR, partner_packages=is_partner_packages
        ).items():
            package = Package(
                package_name, package_path, is_partner=is_partner_packages, upload=False
            )
            package_version = package.get_version(
                package.path, package_name=package.name
            )
            assert package_version
            ver_parts = package_version.split(".")
            # can be 0.0.0.dev0
            assert len(ver_parts) >= 3
            for el in ver_parts[:2]:
                # major and minor version should be int
                assert isinstance(int(el), int)


def test_get_package_source_path() -> None:
    for is_partner_packages in [False, True]:
        for package_name, package_path in get_packages(
            ROOT_DIR, partner_packages=is_partner_packages
        ).items():
            package = Package(
                package_name, package_path, is_partner=is_partner_packages, upload=False
            )
            source_path = package.get_source_path(package.path)
            is_external_package = is_external_repo(package.path)
            if is_external_package:
                assert not source_path
            else:
                assert source_path
                assert source_path.exists()


def test_load_module_members() -> None:
    module_path = "langchain_core.agents"
    ret = load_module_members(module_path)
    assert ret == [
        MemberInfo(
            name="AgentAction",
            qualified_name="langchain_core.agents.AgentAction",
            is_public=True,
            member_type=MemberType.class_,
            class_kind="Pydantic",
        ),
        MemberInfo(
            name="AgentActionMessageLog",
            qualified_name="langchain_core.agents.AgentActionMessageLog",
            is_public=True,
            member_type=MemberType.class_,
            class_kind="Pydantic",
        ),
        MemberInfo(
            name="AgentFinish",
            qualified_name="langchain_core.agents.AgentFinish",
            is_public=True,
            member_type=MemberType.class_,
            class_kind="Pydantic",
        ),
        MemberInfo(
            name="AgentStep",
            qualified_name="langchain_core.agents.AgentStep",
            is_public=True,
            member_type=MemberType.class_,
            class_kind="Pydantic",
        ),
        MemberInfo(
            name="_convert_agent_action_to_messages",
            qualified_name="langchain_core.agents._convert_agent_action_to_messages",
            is_public=False,
            member_type=MemberType.function,
        ),
        MemberInfo(
            name="_convert_agent_observation_to_messages",
            qualified_name="langchain_core.agents._convert_agent_observation_to_messages",
            is_public=False,
            member_type=MemberType.function,
        ),
        MemberInfo(
            name="_create_function_message",
            qualified_name="langchain_core.agents._create_function_message",
            is_public=False,
            member_type=MemberType.function,
        ),
    ]


def test_external_package_load_module_members() -> None:
    module_path = "langchain_google_vertexai.callbacks"
    ret = load_module_members(module_path)
    assert ret == [
        MemberInfo(
            name="VertexAICallbackHandler",
            qualified_name="langchain_google_vertexai.callbacks.VertexAICallbackHandler",
            is_public=True,
            member_type=MemberType.class_,
            class_kind="Regular",
        ),
    ]


def test_get_package_modules() -> None:
    for is_partner_packages in [False, True]:
        for package_name, package_path in get_packages(
            ROOT_DIR, partner_packages=is_partner_packages
        ).items():
            package = Package(
                package_name, package_path, is_partner=is_partner_packages, upload=False
            )
            modules = package.get_modules(package.path)
            assert modules
            first_module = list(modules.values())[0]
            assert isinstance(first_module, Module)
            assert isinstance(first_module.name, str)
            assert isinstance(first_module.namespace, str)
            assert isinstance(first_module.members, list)
            if len(first_module.members) > 0:
                assert isinstance(first_module.members[0], MemberInfo)
            # upload only a single non-partner and a single partner package
            break


def test_get_package_not_parsed() -> None:
    not_parsed_packages = ["partners"]
    for not_required_package in not_parsed_packages:
        packages = get_packages(ROOT_DIR, partner_packages=False)
        assert not_required_package not in packages


def test_package_uploaded() -> None:
    package_name = "langchain-core"
    packages = get_packages(ROOT_DIR, partner_packages=False)
    package_path = packages[package_name]
    package = Package(package_name, package_path, is_partner=False, upload=True)
    assert package.name == package_name
    assert package.path == package_path
    assert package.namespace == "langchain_core"
    assert package.is_partner is False
    assert package.external_repo is False
    assert package.version
    assert package.source_path
    assert package.modules


@pytest.mark.parametrize(
    "package_name,external_repo",
    [
        ("langchain-google-vertexai", True),
        ("langchain-ai21", False),
    ],
)
def test_partner_package_uploaded(package_name: str, external_repo: bool) -> None:
    """The test package should be pip-installed."""
    packages = get_packages(ROOT_DIR, partner_packages=True)
    package_path = packages[package_name]
    package = Package(package_name, package_path, is_partner=True, upload=True)
    assert package.name == package_name
    assert package.path == package_path
    assert package.is_partner is True
    assert package.namespace == package_name.replace("-", "_")
    assert package.external_repo == external_repo
    assert package.version
    if external_repo:
        assert not package.source_path
    else:
        assert package.source_path
    assert package.modules


def test_top_two_levels_of_modules() -> None:
    package_name = "langchain-core"
    packages = get_packages(ROOT_DIR, partner_packages=False)
    package_path = packages[package_name]
    package = Package(package_name, package_path, is_partner=False, upload=True)

    default_members = top_two_levels_of_modules(package)

    with_empty_members = top_two_levels_of_modules(package, remove_empty=False)
    assert len(with_empty_members) > len(default_members)
    assert "langchain_core.__init__" in (set(with_empty_members) - set(default_members))

    no_hidden_members = top_two_levels_of_modules(package, remove_hidden=True)
    assert len(no_hidden_members) < len(default_members)
    assert (set(default_members) - set(no_hidden_members)) == {"langchain_core._api"}
