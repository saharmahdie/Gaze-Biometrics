.PHONY: clean help clean-model test clean clean-data
.DEFAULT_GOAL := help

###########################################################################################################
## VARIABLES
###########################################################################################################

export PYTHON=python3

###########################################################################################################
## GENERAL TARGETS
###########################################################################################################

test: ## run test cases in tests directory
	cd schau_mir_in_die_augen && $(PYTHON) -m unittest discover -t ..

clean: clean-cache ## remove all artifacts
	rm -rf cache/*

clean-cache:
	rm -rf cache/ schau_mir_in_die_augen/.cache/ .cache/ scripts/.cache /tmp/smida-cache

ci-push-image:
	scripts/push_image_ci.sh

# download datasets
dataset-bioeye:
	git submodule update --init --remote -- data/BioEye2015_DevSets
dataset-rigas:
	git submodule update --init --remote -- data/RigasEM
dataset-dyslexia:
	git submodule update --init --remote -- data/Dyslexia
dataset-whl:
	git submodule update --init --remote -- data/where_humans_look

regression-test-train:
	cd scripts && $(PYTHON) train.py --method score-level --dataset bio-tex --clf rbfn
regression-test-eval:
	cd scripts && $(PYTHON) evaluation.py --method score-level --dataset bio-tex --clf rbfn

# jobs for rechecking
eval-score-level-2020:
	cd scripts && $(PYTHON) train.py --method score-level --dataset bio-tex --clf rbfn
	cd scripts && $(PYTHON) evaluation.py --method score-level --dataset bio-tex --clf rbfn
	cd scripts && $(PYTHON) train.py --method score-level --dataset bio-ran --clf rbfn
	cd scripts && $(PYTHON) evaluation.py --method score-level --dataset bio-ran --clf rbfn
	cd scripts && $(PYTHON) train.py --method paper-append --dataset bio-tex --clf rbfn
	cd scripts && $(PYTHON) evaluation.py --method paper-append --dataset bio-tex --clf rbfn
	cd scripts && $(PYTHON) train.py --method paper-append --dataset bio-ran --clf rbfn
	cd scripts && $(PYTHON) evaluation.py --method paper-append --dataset bio-ran --clf rbfn

eval-score-level-2020rf:
	cd scripts && $(PYTHON) train.py --method score-level --dataset bio-tex --clf rf
	cd scripts && $(PYTHON) evaluation.py --method score-level --dataset bio-tex --clf rf
	cd scripts && $(PYTHON) train.py --method score-level --dataset bio-ran --clf rf
	cd scripts && $(PYTHON) evaluation.py --method score-level --dataset bio-ran --clf rf
	cd scripts && $(PYTHON) train.py --method paper-append --dataset bio-tex --clf rf
	cd scripts && $(PYTHON) evaluation.py --method paper-append --dataset bio-tex --clf rf
	cd scripts && $(PYTHON) train.py --method paper-append --dataset bio-ran --clf rf
	cd scripts && $(PYTHON) evaluation.py --method paper-append --dataset bio-ran --clf rf
