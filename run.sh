#!/bin/bash

read input_path output_path 
while getopts 'ndcb' OPTION; do
  case "$OPTION" in
    n)
        python main.py $input_path $output_path "0"
        ;;
    d)
        python main.py $input_path $output_path "1"
        ;;
    c)
        python main.py $input_path $output_path "2"
        ;;
    b)
        python main.py $input_path $output_path "3"
      ;;
  esac
done
shift "$(($OPTIND -1))"