# Recursive GNU Make wildcard implementation
# https://stackoverflow.com/questions/2483182/recursive-wildcards-in-gnu-make/18258352#18258352
rwildcard=$(foreach d,$(wildcard $(1:=/*)),$(call rwildcard,$d,$2) $(filter $(subst *,%,$2),$d))

format: ## [Local development] Auto-format python code using black
	black src

install: ## [Local development] Install packages
	python -m venv .env
	source .env/bin/activate
	pip install -r requirements.txt
