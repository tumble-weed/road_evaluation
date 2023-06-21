#!/bin/bash
# Call this script from the experiments/explanation_generation folder as cwd.
#methods=("ig" "gb" "ig_sg" "gb_sg" "ig_sq" "gb_sq" "ig_var" "gb_var")
methods=("gb" "ig" "ig_sg" "gb_sg" "ig_sq" "gb_sq" "ig_var" "gb_var")
for method in ${methods[@]}; do
echo "$method"
python ExplanationGeneration.py --expl_method="${method}"
done

