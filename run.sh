# run below with wsl2
# or run by: ./run.sh

jupyter-book build .
cp -TRv _build/html docs/ > /dev/null # overwrite https://stackoverflow.com/questions/23698183/how-to-force-cp-to-overwrite-directory-instead-of-creating-another-one-inside
cp ./.nojekyll ./docs/.nojekyll 