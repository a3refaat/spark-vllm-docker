# Model download management
#
# Usage (run as the vllm user on spark1):
#   make help              — list all per-recipe download targets
#   make status            — show download state of all recipe models
#   make download-missing  — download any missing/incomplete models
#   make download-all      — (re-)download every recipe model
#   make download-<slug>   — download the specific recipe (e.g. make download-minimax-m2.7-nvfp4-gb10)
#
# Requires: python3, pyyaml, uvx (installed by Ansible to /usr/local/bin)
#
# Run as vllm:
#   sudo -u vllm make status
#   sudo -u vllm make download-missing
#   sudo -u vllm make download-minimax-m2.7-nvfp4-gb10

HF_HOME  ?= $(HOME)/.cache/huggingface
HUB       = $(HF_HOME)/hub
RECIPES   = $(wildcard recipes/*.yaml)

# Extract model checkpoints from all recipe YAML files (for bulk targets)
MODELS := $(shell python3 -c \
  "import yaml, glob; \
   [print(d['model']) for f in sorted(glob.glob('recipes/*.yaml')) \
    if (d := yaml.safe_load(open(f))) and 'model' in d]" \
  2>/dev/null)

# Recipe slugs = YAML filenames without .yaml extension
RECIPE_SLUGS := $(patsubst recipes/%.yaml,%,$(sort $(wildcard recipes/*.yaml)))

# Generate a download-<slug> target for each recipe.
# The model checkpoint is read from the YAML at target execution time (not load
# time) to avoid one python3 call per recipe at make startup.
#
# Dollar-sign escaping in define/eval:
#   $$$$VAR in define → $$VAR after $(call) → $VAR in shell
define RECIPE_DOWNLOAD_RULE
.PHONY: download-$(1)
download-$(1):
	@MODEL=$$$$(python3 -c "import yaml; d=yaml.safe_load(open('recipes/$(1).yaml')); print(d.get('model',''))" 2>/dev/null); \
	[ -n "$$$$MODEL" ] || { echo "Error: no model field in recipes/$(1).yaml"; exit 1; }; \
	./hf-download.sh "$$$$MODEL" -c --copy-parallel

endef

$(foreach slug,$(RECIPE_SLUGS),$(eval $(call RECIPE_DOWNLOAD_RULE,$(slug))))

.PHONY: help status download-missing download-all

help:
	@echo "Per-recipe download targets:"
	@$(foreach slug,$(RECIPE_SLUGS),echo "  make download-$(slug)";)
	@echo ""
	@echo "Bulk targets:"
	@echo "  make status           — show download state of all models"
	@echo "  make download-missing — download only missing/incomplete models"
	@echo "  make download-all     — (re-)download every model"

status:
	@if [ -z "$(RECIPE_SLUGS)" ]; then echo "No recipes found in recipes/"; exit 1; fi
	@printf "%-12s  %-42s  %s\n" "STATUS" "RECIPE" "MODEL"
	@printf "%-12s  %-42s  %s\n" "------" "------" "-----"
	@for f in $(sort $(RECIPES)); do \
	  slug=$$(basename "$$f" .yaml); \
	  m=$$(python3 -c "import yaml; d=yaml.safe_load(open('$$f')); print(d.get('model','unknown'))" 2>/dev/null); \
	  dir=$(HUB)/models--$$(echo "$$m" | sed 's|/|--|g'); \
	  if   [ ! -d "$$dir/blobs" ]; then \
	    printf "%-12s  %-42s  %s\n" "MISSING" "$$slug" "$$m"; \
	  elif ls "$$dir/blobs/"*.incomplete 2>/dev/null | grep -q .; then \
	    count=$$(ls "$$dir/blobs/"*.incomplete 2>/dev/null | wc -l); \
	    printf "%-12s  %-42s  %s  (%d incomplete)\n" "INCOMPLETE" "$$slug" "$$m" "$$count"; \
	  else \
	    printf "%-12s  %-42s  %s\n" "COMPLETE" "$$slug" "$$m"; \
	  fi; \
	done

download-missing:
	@if [ -z "$(MODELS)" ]; then echo "No models found in recipes/"; exit 1; fi
	@for m in $(MODELS); do \
	  dir=$(HUB)/models--$$(echo "$$m" | sed 's|/|--|g'); \
	  if [ ! -d "$$dir/blobs" ] || ls "$$dir/blobs/"*.incomplete 2>/dev/null | grep -q .; then \
	    echo "==> Downloading: $$m"; \
	    ./hf-download.sh "$$m" -c --copy-parallel || exit 1; \
	  else \
	    echo "SKIP (complete): $$m"; \
	  fi; \
	done
	@echo "All missing models downloaded."

download-all:
	@if [ -z "$(MODELS)" ]; then echo "No models found in recipes/"; exit 1; fi
	@for m in $(MODELS); do \
	  echo "==> $$m"; \
	  ./hf-download.sh "$$m" -c --copy-parallel || exit 1; \
	done
	@echo "All models downloaded."
