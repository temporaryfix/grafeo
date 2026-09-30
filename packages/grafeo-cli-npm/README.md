# @grafeo-db/cli

Command-line interface for [Grafeo](https://grafeo.dev) graph database, distributed via npm.

## Installation

```bash
npm install -g @grafeo-db/cli
# or
npx @grafeo-db/cli --help
```

The correct platform-specific binary is installed automatically via optional dependencies.
If that dependency is missing or incomplete, the launcher reports an error.
Reinstall with optional dependencies enabled; the launcher requires its matching
native package even when another `grafeo` executable is installed.

## Supported Platforms

| Platform | Package |
|----------|---------|
| Linux x64 | `@grafeo-db/cli-linux-x64` |
| Linux ARM64 | `@grafeo-db/cli-linux-arm64` |
| macOS x64 | `@grafeo-db/cli-darwin-x64` |
| macOS ARM64 | `@grafeo-db/cli-darwin-arm64` |
| Windows x64 | `@grafeo-db/cli-win32-x64` |

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
```

## Alternative Installation

```bash
# Via Rust
cargo install grafeo-cli

# Via pip/uv
uv add grafeo-cli

# Direct download
# https://github.com/GrafeoDB/grafeo/releases
```

## Links

- [Documentation](https://grafeo.dev)
- [GitHub](https://github.com/GrafeoDB/grafeo)

## License

Apache-2.0
