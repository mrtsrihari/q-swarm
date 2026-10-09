"""Test CLI and basic imports."""

from qswarm.cli import build_parser, validate_environment


def test_cli_parser_build():
    parser = build_parser()
    assert parser.prog == "qswarm"
    args = parser.parse_args(["validate"])
    assert args.subcommand == "validate"


def test_cli_validate():
    code = validate_environment(verbose=False)
    assert code == 0


def test_version_import():
    import qswarm
    assert qswarm.__version__ == "0.1.0"
