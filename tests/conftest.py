"""Shared pytest fixtures for the magform test-suite."""

from __future__ import annotations

import pytest

from magform import materials


@pytest.fixture
def cu():
    return materials.get("copper_ofhc")


@pytest.fixture
def w():
    return materials.get("tungsten")


@pytest.fixture
def tial():
    return materials.get("gamma_tial")


@pytest.fixture
def al():
    return materials.get("aluminum_6061")


@pytest.fixture
def ti():
    return materials.get("titanium_cp2")


@pytest.fixture
def steel():
    return materials.get("steel_1018")
