# Claude Code Configuration

## Virtual Environment
This project uses `uv` for dependency management. Use the `uva` command to activate the virtual environment before running any Python/MkDocs commands.

## Dependencies
- MkDocs with Material theme: `uv pip install mkdocs-material`

## Commands
- Activate environment: `uva`
- Serve MkDocs: `mkdocs serve -a localhost:7282`
- Build MkDocs: `mkdocs build`

## Project Structure
- `/docs/*.md` - Documentation files (pure markdown only)
- `mkdocs.yml` - MkDocs configuration with Material theme, dark mode, and left navigation
- No CSS/JS files - using pure markdown with Material theme defaults