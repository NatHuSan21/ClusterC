search_dir="$(pwd)"
for entry in $(ls $search_dir)
do
	echo "Adding $entry"
	git add $entry
	git commit -m "Add $entry"
	git push
done
