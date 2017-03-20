#!/bin/bash
for (( i = 0; i < 10; i++ )); do
	python ga_artigo.py dados/test_90_$i.csv
	# echo "$i"
done
