.PHONY: install style test

PYTHON := python
CHECK_DIRS := minference tests experiments examples
EXELUDE_DIRS := minference/kernels

install:
	@${PYTHON} setup.py bdist_wheel
	@${PYTHON} -m pip install dist/sdtools*

style:
	black $(CHECK_DIRS) --exclude ${EXELUDE_DIRS}
	isort -rc $(CHECK_DIRS)

test:
	@${PYTHON} -m pytest -n auto --dist=loadfile -s -v ./tests/
