# grafeo-cli

Command-line interface for [Grafeo](https://grafeo.dev) graph database.

This is a thin Python launcher package that runs the pre-built Grafeo CLI binary.

## Installation

```bash
uv add grafeo-cli
# or: pip install grafeo-cli
```

The launcher requires the `grafeo` executable bundled with its platform-specific
wheel. If it is missing, reinstall the wheel for your platform.

The source distribution includes the Rust workspace and compiles the executable
during installation. Building from source requires Rust 1.91.1 or newer, Cargo,
a native C/C++ toolchain and enough temporary disk space for a release build.
Cargo uses the included lockfile; dependencies must be available locally or
downloadable. The resulting wheel bundles the executable, and temporary native
build outputs are removed afterward.

You can also install and run the native CLI directly:

```bash
# Via cargo
cargo install grafeo-cli

# Or download from GitHub releases
# https://github.com/GrafeoDB/grafeo/releases
```

## Usage

```bash
# Database management
grafeo info ./my-db
grafeo stats ./my-db
grafeo validate ./my-db

# Query execution
grafeo query ./my-db "MATCH (n:Person) RETURN n.name"

# Interactive shell
grafeo shell ./my-db

# Create a new database
grafeo init ./new-db

# Shell completions
grafeo completions bash > ~/.local/share/bash-completion/completions/grafeo
```

## Links

- [Documentation](https://grafeo.dev)
- [GitHub](https://github.com/GrafeoDB/grafeo)
- [Grafeo Python Library](https://pypi.org/project/grafeo/) (the database engine, not this CLI tool)

## License

Apache-2.0
