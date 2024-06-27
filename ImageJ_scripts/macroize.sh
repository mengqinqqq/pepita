#!/bin/bash

# Take .ijm macro files and aggregate them into a single macro file to place into ImageJ dir automatically.
# Takes arguments in pairs: an IJM file and a hotkey to use to run it in ImageJ; any number of pairs.
# Use like:
# ./macroize.sh OpenForMasking.ijm f3 OpenForMaskingSingle.ijm f4 >> "/c/Program Files/ImageJ/macros/StartupMacros.txt"

aggregate=""
nl=$'\n'

# for ((i=1; i <= "$#"; i++))
for i in $(seq 1 2 $#)
do
	j=$((i+1))
	new_macro="macro \"${!i/.ijm/} [${!j}]\" {${nl}$(sed -e 's/^/    /' ${!i})${nl}}"
	# explanation of the contents of the above string:
	# - literal "macro"
	# - literal space and double quote
	# - the script argument represented by number $i, with ".ijm" removed from it
	# - literal space
	# - the script argument represented by number $j, surrounded by literal brackets
	# - literal double quote, space, and opening curly brace
	# - newline
	# - the contents of the file named by the script argument represented by the number $i, with 4 spaces prepended to every line
	# - newline
	# - literal closing curly brace

	aggregate="${aggregate}${nl}${nl}${new_macro}"
	# echo "${new_macro}${nl}"
done

echo "==== CUSTOM MACROS ====${nl}$aggregate"
