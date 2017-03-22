#!/bin/bash
l="10 30 50 70 90"
for k in $l; do
	for (( i = 0; i < 10; i++ )); do
		# python ga_artigo.py $i $k
		python ga_artigo.py dados_new/test-$k-$i-adaptativo-new.csv $k
		# echo "$i $k"
	done
	echo ""
done