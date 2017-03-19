#!/bin/bash
for (( i = 0; i < 10; i++ )); do
	python ga_artigo.py $i
	echo "$i"
done