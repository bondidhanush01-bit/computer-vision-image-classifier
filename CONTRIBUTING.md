# Contributing to Computer Vision Image Classifier

Thank you for your interest in contributing to this project! We welcome contributions from the community.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- A clear, descriptive title
- A detailed description of the bug
- Steps to reproduce the issue
- Expected vs. actual behavior
- Your environment (Python version, OS, etc.)
- Relevant code snippets or error messages

### Suggesting Enhancements

For feature requests:
- Use a clear, descriptive title
- Provide a detailed description of the proposed enhancement
- List examples of how the feature would work
- Explain why this enhancement would be useful

### Pull Requests

1. **Fork the repository** and create your branch from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the existing code style
   - Write clear, descriptive commit messages
   - Add or update tests as needed
   - Update documentation if required

3. **Test your changes**
   ```bash
   pytest tests/ -v
   pytest tests/ --cov=src  # With coverage
   ```

4. **Lint your code**
   ```bash
   flake8 src/ tests/
   black src/ tests/
   mypy src/
   ```

5. **Push to your fork** and submit a pull request
   - Provide a clear description of the changes
   - Link any relevant issues
   - Ensure all CI checks pass

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/bondidhanush01-bit/computer-vision-image-classifier.git
   cd computer-vision-image-classifier
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Write descriptive docstrings
- Keep functions focused and modular
- Maximum line length: 120 characters

### Example Code Style:

```python
def process_image(image_path: str, model: torch.nn.Module) -> Dict[str, float]:
    """
    Process an image and return predictions.
    
    Args:
        image_path: Path to the image file
        model: Trained PyTorch model
    
    Returns:
        Dictionary with class predictions and confidence scores
    """
    image = load_image(image_path)
    predictions = model(image)
    return format_predictions(predictions)
```

## Testing Guidelines

- Write tests for new features
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage
- Use pytest conventions

```python
def test_feature_works():
    """Test that the feature works correctly."""
    result = feature_function(test_input)
    assert result == expected_output
```

## Documentation

- Update README.md for major changes
- Add docstrings to all functions
- Update CHANGELOG.md with notable changes
- Include examples in docstrings

## Commit Messages

Use clear, descriptive commit messages:
- ✅ Good: `Add image augmentation to preprocessing pipeline`
- ❌ Bad: `fix bug`, `update`

Format:
```
[Type] Brief description

Longer explanation if needed.
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

## Code Review Process

1. A maintainer will review your PR
2. Address any feedback or requested changes
3. Re-request review once changes are made
4. Once approved, your PR will be merged

## Questions?

- Check existing issues and PRs
- Read the README.md
- Open a discussion
- Contact via LinkedIn: [Dhanush Bondi](https://www.linkedin.com/in/dhanush-bondi-978697352/)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing! 🙏
