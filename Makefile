.PHONY: install style test

PYTHON := python
CHECK_DIRS := minference tests experiments examples
EXELUDE_DIRS := minference/ops
FORCE_EXELUDE_DIRS := minference/modules/minference_forward.py

install:
	@${PYTHON} setup.py bdist_wheel
	@${PYTHON} -m pip install dist/minference*

style:
	black $(CHECK_DIRS) --extend-exclude ${EXELUDE_DIRS} --force-exclude ${FORCE_EXELUDE_DIRS}
	isort -rc $(CHECK_DIRS)

test:
	@${PYTHON} -m pytest -n auto --dist=loadfile -s -v ./tests/
