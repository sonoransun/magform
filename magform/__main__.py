"""Allow ``python -m magform ...`` as an alias for the ``magform`` console script."""

from .cli import main

raise SystemExit(main())
