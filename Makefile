init:
	pip install --upgrade -r requirements.txt

test:
	py.test tests

.PHONY: init test
