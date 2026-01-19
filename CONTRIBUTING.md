# Contributing to Veo-style Soccer Match Analysis

Thank you for your interest in contributing! This project follows a "vibe coding" philosophy with heavy emphasis on caching, end-to-end testing, and quality over speed.

## Development Philosophy

See `AGENTS.md` for detailed architectural principles. Key points:

- **Quality first**: Accuracy and robustness over raw speed
- **Local-first**: Apple Silicon optimization with MPS
- **End-to-end slices**: Complete features over isolated components
- **Cache everything**: Support fast iteration loops
- **Proven patterns first**: Duplicate successful approaches before inventing new ones

## Getting Started

1. **Fork and clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/ai_video_analysis.git
   cd ai_video_analysis
   ```

2. **Set up environment**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

3. **Verify setup**
   ```bash
   python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
   pytest tests/ -v
   ```

## Development Workflow

### 1. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 2. Make Changes

Follow the project structure:
- `src/` - Source code organized by pipeline stages
- `tests/` - Unit and integration tests
- `configs/` - YAML configurations
- `docs/` - Feature specs and decision docs

### 3. Code Style

We use:
- **Black** for formatting
- **Ruff** for linting
- **MyPy** for type checking

```bash
# Format code
black src/ tests/

# Check linting
ruff check src/ tests/

# Type check
mypy src/
```

### 4. Testing

All new features must include tests:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_detection.py -v
```

**Test Requirements:**
- Unit tests for individual functions/classes
- Integration tests for pipeline stages
- At least one end-to-end test on sample video
- Golden tests for critical output formats

### 5. Commit Messages

Use clear, descriptive commit messages:

```
feat: add ByteTrack multi-object tracking
fix: handle missing ball detections gracefully
docs: update README with performance benchmarks
test: add integration test for team clustering
refactor: extract color clustering to separate module
```

Prefixes:
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation only
- `test:` - Test additions/changes
- `refactor:` - Code changes without behavior change
- `perf:` - Performance improvements
- `chore:` - Maintenance tasks

### 6. Documentation

Update documentation for:
- New features → README.md and docstrings
- Configuration changes → Update config schema and README
- API changes → Update AGENTS.md
- Breaking changes → Migration guide in PR description

## Artifact Requirements

Every pipeline stage must produce versioned artifacts:

```python
{
    "schema_version": "1.0",
    # ... data
}
```

When changing output format:
- Bump schema version
- Add migration logic
- Update tests
- Document in PR

## Pull Request Process

1. **Before submitting:**
   - [ ] Tests pass: `pytest tests/ -v`
   - [ ] Code formatted: `black src/ tests/`
   - [ ] Linting clean: `ruff check src/ tests/`
   - [ ] Documentation updated
   - [ ] CHANGELOG entry added (if applicable)

2. **PR Template:**
   - Fill out all sections in `.github/PULL_REQUEST_TEMPLATE.md`
   - Link related issues
   - Include performance impact notes
   - Add screenshots/output samples if relevant

3. **Review Process:**
   - Maintainers will review within 1-2 weeks
   - Address feedback with additional commits
   - Once approved, squash-merge to main

## Performance Benchmarks

When adding performance-impacting features:

1. **Benchmark before/after** on reference videos
2. **Include timing data** in PR description
3. **Test on both MPS and CPU** if device-related
4. **Document trade-offs** (speed vs accuracy)

Example:
```
YOLOv8m detection on M1 Air:
- Before: 350ms/frame
- After: 320ms/frame (8% faster)
- Tested on 1920x1080 @ 30fps
```

## Milestone Priorities

Current focus areas:

1. **Milestone 2** (Next): Multi-object tracking
   - ByteTrack implementation
   - Track ID stability
   - Quality metrics

2. **Milestone 3**: Team identification
   - Jersey color clustering
   - Team assignment logic

See README.md roadmap for full milestone list.

## Questions?

- Open an issue for bugs or feature requests
- Check existing issues before creating duplicates
- Tag with appropriate labels (bug, enhancement, documentation, etc.)

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Assume good intentions
- Help others learn and grow

---

**Remember**: This is a research project with emphasis on quality and reproducibility. Take time to do it right! 🎯
