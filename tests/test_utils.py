"""
Unit tests for utility functions
"""

import pytest
import torch
from src.utils import get_device, create_directories
from pathlib import Path
import tempfile
import shutil


class TestUtilityFunctions:
    """Test cases for utility functions"""
    
    def test_get_device(self):
        """Test device selection"""
        device = get_device()
        assert device in ['cuda', 'cpu']
        assert isinstance(device, str)
    
    def test_create_directories(self):
        """Test directory creation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dirs = [
                str(Path(tmpdir) / 'test1'),
                str(Path(tmpdir) / 'test2' / 'nested'),
            ]
            create_directories(test_dirs)
            
            for dir_path in test_dirs:
                assert Path(dir_path).exists()
    
    def test_create_directories_empty_list(self):
        """Test create directories with empty list"""
        # Should not raise any error
        create_directories([])
    
    def test_create_directories_existing(self):
        """Test creating existing directories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should not raise error if directory already exists
            create_directories([tmpdir])
            assert Path(tmpdir).exists()
