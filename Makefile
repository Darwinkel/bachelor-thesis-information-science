folder?=code

check:
	isort $(folder)
	black $(folder)
	vulture $(folder)
	mypy $(folder)
	pylint --recursive=yes $(folder)