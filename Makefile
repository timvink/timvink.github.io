install:
	uv export -o requirements.txt --no-hashes

docs:
	uv run mkdocs serve