PY := .venv/bin/python
LIMIT ?=

.PHONY: help install test smoke sample label label-stats relabel relabel-stats train retrain \
        api worker worker-once ask dashboard up up-eval down logs loadtest docker-retrain

help:  ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ---------- setup & tests ----------

install:  ## Create the virtualenv and install everything (dev + dashboard + data)
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev,dashboard,data]"

test:  ## Run the test suite (every LLM call is mocked)
	$(PY) -m pytest

smoke:  ## Phase 1 baseline: same prompts to every tier (make smoke LIMIT=3)
	$(PY) scripts/smoke_test.py $(if $(LIMIT),--limit $(LIMIT),)

# ---------- data & classifier ----------

sample:  ## Sample 300 labeling candidates + 1,000 load-test prompts from Dolly 15k
	$(PY) scripts/sample_prompts.py

label:  ## Label candidates, round 1 (make label LIMIT=25)
	$(PY) scripts/label_prompts.py $(if $(LIMIT),--limit $(LIMIT),)

label-stats:  ## Round-1 labeling progress
	$(PY) scripts/label_prompts.py --stats

relabel:  ## Labeling round 2 with the written rules (make relabel LIMIT=50)
	$(PY) scripts/label_prompts.py --round 2 $(if $(LIMIT),--limit $(LIMIT),)

relabel-stats:  ## Round-2 progress + agreement with round 1
	$(PY) scripts/label_prompts.py --round 2 --stats

train:  ## Train the classifier; new version goes live only if not worse
	$(PY) scripts/train_classifier.py

retrain:  ## Feedback loop on the host DB: human labels + routing failures
	$(PY) scripts/retrain.py

# ---------- run locally (no Docker) ----------

api:  ## FastAPI server on :8000 (docs at /docs)
	$(PY) -m uvicorn autopilot.api.app:app --host 0.0.0.0 --port 8000

worker:  ## Background verification worker (until Ctrl+C)
	$(PY) scripts/worker.py

worker-once:  ## Verify everything queued, then exit
	$(PY) scripts/worker.py --once

ask:  ## Send N load-test prompts through the autopilot in-process (make ask LIMIT=10)
	$(PY) scripts/ask.py --loadtest $(or $(LIMIT),5)

dashboard:  ## Streamlit dashboard on :8501 (reads AUTOPILOT_DB_PATH or data/autopilot.db)
	$(PY) -m streamlit run src/autopilot/dashboard/app.py

# ---------- Docker ----------

up:  ## Start api + worker + dashboard in Docker (normal mode: 15% verified)
	docker compose up -d --build

up-eval:  ## Same, in evaluation mode (verify every request) for load tests
	AUTOPILOT_MODE=evaluation WORKER_CONCURRENCY=6 docker compose up -d --build

down:  ## Stop the Docker stack (the database volume is kept)
	docker compose down

logs:  ## Follow api + worker logs
	docker compose logs -f api worker

loadtest:  ## Load test against the running stack (make up-eval first; make loadtest LIMIT=500)
	$(PY) scripts/loadtest.py --limit $(or $(LIMIT),1000)

docker-retrain:  ## Feedback loop using the Docker database
	docker compose run --rm worker python scripts/retrain.py
