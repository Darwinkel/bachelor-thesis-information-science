folder?=code

check:
	isort $(folder)
	black $(folder)
	vulture $(folder)
	eradicate -r $(folder)
	pylint --recursive=yes $(folder)
	mypy $(folder)