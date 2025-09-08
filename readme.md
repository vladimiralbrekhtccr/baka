# setup
mkdir baka
cd baka
uv venv --python 3.12 .venv
uva # alias uva='source .venv/bin/activate'
code .gitgnore
code readme.nd
uv pip install mkdocs
mkdocs new . 
mkdocs serve -a localhost:7775

# git
git add . 
git commit -m "baka"
git remote add origin git@github.com:vladimiralbrekhtccr/baka.git
git branch -M main
git push -u origin main