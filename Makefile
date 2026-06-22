# Model download management
#
# Usage (run as the vllm user on spark1):
#   make status            — show download state of all recipe models
#   make download-missing  — download any missing/incomplete models
#   make download-all      — (re-)download every recipe model
#
# Requires: python3, pyyaml, uvx (installed by Ansible into /opt/uv)
#
# Run as vllm:
#   sudo -u vllm make status
#   sudo -u vllm make download-missing

HF_HOME  ?= $(HOME)/.cache/huggingface
HUB       = $(HF_HOME)/hub
RECIPES   = $(wildcard recipes/*.yaml)

# Extract model checkpoints from all recipe YAML files
MODELS := $(shell python3 -c \
  "import yaml, glob; \
   [print(d['model']) for f in sorted(glob.glob('recipes/*.yaml')) \
    if (d := yaml.safe_load(open(f))) and 'model' in d]" \
  2>/dev/null)

.PHONY: status download-missing download-all

status:
	@if [ -z "$(MODELS)" ]; then echo "No models found in recipes/"; exit 1; fi
	@printf "%-12s  %s\n" "STATUS" "MODEL"
	@printf "%-12s  %s\n" "------" "-----"
	@for m in $(MODELS); do \
	  dir=$(HUB)/models--$$(echo "$$m" | sed 's|/|--|g'); \
	  if   [ ! -d "$$dir/blobs" ]; then \
	    printf "%-12s  %s\n" "MISSING" "$$m"; \
	  elif ls "$$dir/blobs/"*.incomplete 2>/dev/null | grep -q .; then \
	    count=$$(ls "$$dir/blobs/"*.incomplete 2>/dev/null | wc -l); \
	    printf "%-12s  %s  (%d incomplete)\n" "INCOMPLETE" "$$m" "$$count"; \
	  else \
	    printf "%-12s  %s\n" "COMPLETE" "$$m"; \
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
